"""Level 1 builder: generate domain shelves from document cards.

Groups document cards by domain, asks the LLM to cluster them into
thematic groups, and produces a domain-level overview.
"""

import json
import re
from pathlib import Path

from loguru import logger

from config.domains import DOMAINS
from data_pipeline.hierarchy.hierarchy_models import (
    DocumentCard,
    DocumentCardBrief,
    DocumentGroup,
    DomainShelf,
)
from llm.claude_client import ClaudeClient
from llm.gemini_client import GeminiClient
from llm.ollama_client import OllamaClient

_DOMAIN_SHELF_PROMPT = """You are building a domain-level summary for a section of an insurance company knowledge base.

Domain: {domain} ({domain_he})
Number of documents: {doc_count}

Document summaries:
---
{documents_text}
---

Generate a JSON object with:
1. "overview": A 3-5 sentence overview of this insurance domain — what types of coverage it includes, who needs it, what kinds of questions and topics are covered across all documents.
2. "document_groups": A list of thematic groups that cluster related documents together. Each group has:
   - "group_name": A short descriptive name (e.g., "Policy Terms & Conditions", "Claims & Procedures", "FAQ & General Info")
   - "group_summary": 1-2 sentences describing what this group of documents covers together
   - "doc_ids": List of doc_id values belonging to this group
   Every document must appear in exactly one group. Create 2-6 groups depending on the domain size.

Respond with ONLY the JSON object, no markdown fences."""


def _group_cards_by_domain(cards: list[DocumentCard]) -> dict[str, list[DocumentCard]]:
    """Group document cards by domain."""
    groups: dict[str, list[DocumentCard]] = {}
    for card in cards:
        groups.setdefault(card.domain, []).append(card)
    return groups


def build_domain_shelves(
    cards: list[DocumentCard],
    output_dir: Path = Path("data/hierarchy"),
    claude_client: ClaudeClient | None = None,
    gemini_client: GeminiClient | None = None,
    ollama_client: OllamaClient | None = None,
    llm_mode: str = "auto",
) -> list[DomainShelf]:
    """Build domain shelves from document cards.

    Args:
        cards: All document cards from Level 2.
        output_dir: Where to write domain JSON files.
        claude_client: Pre-created Claude client.
        gemini_client: Pre-created Gemini client.
        ollama_client: Pre-created Ollama client.
        llm_mode: "claude", "gemini", "ollama", or "auto".

    Returns:
        List of DomainShelf objects.
    """
    if llm_mode in ("claude", "auto") and claude_client is None:
        try:
            claude_client = ClaudeClient()
        except Exception as e:
            logger.warning(f"Failed to create Claude client: {e}")

    if llm_mode in ("gemini", "auto") and gemini_client is None:
        try:
            gemini_client = GeminiClient()
        except Exception as e:
            logger.warning(f"Failed to create Gemini client: {e}")

    if llm_mode in ("ollama", "auto") and ollama_client is None:
        try:
            ollama_client = OllamaClient()
            if not ollama_client.is_available():
                ollama_client = None
        except Exception:
            pass

    groups = _group_cards_by_domain(cards)
    logger.info(f"Building domain shelves for {len(groups)} domains...")

    shelves: list[DomainShelf] = []

    for domain_name, domain_cards in groups.items():
        domain_meta = DOMAINS.get(domain_name)
        domain_he = domain_meta.name_he if domain_meta else domain_name

        # Build documents text for prompt
        doc_parts = []
        for c in domain_cards:
            parts = [f"doc_id: {c.doc_id} | title: {c.title} (type: {c.doc_type})"]
            if c.summary:
                parts.append(f"  Summary: {c.summary[:300]}")
            if c.key_topics:
                parts.append(f"  Topics: {', '.join(c.key_topics[:8])}")
            doc_parts.append("\n".join(parts))
        documents_text = "\n\n".join(doc_parts)[:12000]

        prompt = _DOMAIN_SHELF_PROMPT.format(
            domain=domain_name,
            domain_he=domain_he,
            doc_count=len(domain_cards),
            documents_text=documents_text,
        )

        result = None

        # Try Claude
        if claude_client and llm_mode in ("claude", "auto"):
            try:
                response = claude_client.generate(
                    prompt, temperature=0.0, max_tokens=2048
                )
                response = response.strip()
                response = re.sub(r"^```(?:json)?\s*", "", response)
                response = re.sub(r"\s*```$", "", response)
                result = json.loads(response)
            except Exception as e:
                logger.warning(f"Claude failed for domain {domain_name}: {e}")

        # Try Gemini
        if result is None and gemini_client and llm_mode in ("gemini", "auto"):
            try:
                response = gemini_client.generate(
                    prompt, temperature=0.0, max_tokens=2048
                )
                response = response.strip()
                response = re.sub(r"^```(?:json)?\s*", "", response)
                response = re.sub(r"\s*```$", "", response)
                result = json.loads(response)
            except Exception as e:
                logger.warning(f"Gemini failed for domain {domain_name}: {e}")

        # Fallback to Ollama
        if result is None and ollama_client and llm_mode in ("ollama", "auto"):
            try:
                response = ollama_client.generate(
                    prompt, temperature=0.0, max_tokens=2048
                )
                response = response.strip()
                response = re.sub(r"^```(?:json)?\s*", "", response)
                response = re.sub(r"\s*```$", "", response)
                result = json.loads(response)
            except Exception as e:
                logger.warning(f"Ollama failed for domain {domain_name}: {e}")

        if result is None:
            result = {"overview": "", "document_groups": []}

        # Parse document groups
        known_doc_ids = {c.doc_id for c in domain_cards}
        doc_groups = []
        grouped_ids: set[str] = set()
        for g in result.get("document_groups", []):
            valid_ids = [d for d in g.get("doc_ids", []) if d in known_doc_ids]
            if valid_ids:
                doc_groups.append(
                    DocumentGroup(
                        group_name=g.get("group_name", "Other"),
                        group_summary=g.get("group_summary", ""),
                        doc_ids=valid_ids,
                    )
                )
                grouped_ids.update(valid_ids)

        # Add ungrouped docs to an "Other" group
        ungrouped = [c.doc_id for c in domain_cards if c.doc_id not in grouped_ids]
        if ungrouped:
            doc_groups.append(
                DocumentGroup(
                    group_name="Other",
                    group_summary="Additional documents.",
                    doc_ids=ungrouped,
                )
            )

        # Build brief entries
        all_docs_brief = [
            DocumentCardBrief(
                doc_id=c.doc_id,
                title=c.title,
                doc_type=c.doc_type,
                summary=c.summary[:200] if c.summary else "",
                key_topics=c.key_topics[:5],
            )
            for c in domain_cards
        ]

        total_chunks = sum(c.chunk_count for c in domain_cards)

        shelf = DomainShelf(
            domain=domain_name,
            domain_he=domain_he,
            overview=result.get("overview", ""),
            document_groups=doc_groups,
            all_documents=all_docs_brief,
            total_documents=len(domain_cards),
            total_chunks=total_chunks,
        )
        shelves.append(shelf)

        # Write to disk
        domain_dir = output_dir / "domains"
        domain_dir.mkdir(parents=True, exist_ok=True)
        (domain_dir / f"{domain_name}.json").write_text(
            shelf.model_dump_json(indent=2), encoding="utf-8"
        )

    logger.info(f"Built {len(shelves)} domain shelves")
    return shelves
