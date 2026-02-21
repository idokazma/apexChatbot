"""Level 2 builder: generate document cards from parsed documents.

Reads the full parsed markdown for each document and asks the LLM to
produce a rich structured summary (DocumentCard). Supports checkpoint/
resume and dual-LLM (Claude + Ollama fallback).
"""

import json
import re
from pathlib import Path

from loguru import logger

from data_pipeline.chunker.chunk_models import Chunk
from data_pipeline.hierarchy.hierarchy_models import DocumentCard
from llm.claude_client import ClaudeClient
from llm.gemini_client import GeminiClient
from llm.ollama_client import OllamaClient

_DOCUMENT_CARD_PROMPT = """You are building a document summary card for a search system used by another LLM.

The LLM reader will use this card to decide: "Does this document contain the information I need?"

Document title: {title}
Domain: {domain}
Source URL: {source_url}
Document type: {doc_type}

Full document content (may be truncated):
---
{content}
---

Generate a JSON object with these fields:
1. "summary": A 5-10 sentence overview written as notes for another LLM. Describe what this document is about, what topics it covers, what questions it can answer. Be specific — mention policy names, coverage types, conditions, exclusions, amounts. Example style: "This document contains information about X, Y, Z. It covers policy terms for A, exclusion conditions for B, pricing details for C. A reader looking for D or E would find relevant information here."
2. "key_topics": A list of 10-20 specific topics or questions this document can answer (in the same language as the content). Be specific, not generic.
3. "key_facts": A list of important numbers, conditions, exclusions, dates, or amounts mentioned in this document. Empty list if none found.
4. "document_type_note": A short note about what kind of document this is, e.g., "This is a full insurance policy document with terms and conditions" or "This is an FAQ page answering common questions" or "This is a marketing page with general information".

Respond with ONLY the JSON object, no markdown fences or extra text."""


def _load_parsed_documents(parsed_dir: Path) -> list[dict]:
    """Load all parsed documents from disk.

    Returns list of dicts with keys: doc_id, title, domain, source_url,
    file_type, markdown.
    """
    docs = []
    for domain_dir in sorted(parsed_dir.iterdir()):
        if not domain_dir.is_dir():
            continue
        domain = domain_dir.name
        for doc_path in sorted(domain_dir.glob("*.json")):
            try:
                data = json.loads(doc_path.read_text(encoding="utf-8"))
                docs.append({
                    "doc_id": doc_path.stem,
                    "title": data.get("title", doc_path.stem),
                    "domain": data.get("domain", domain),
                    "source_url": data.get("source_url", ""),
                    "file_type": data.get("file_type", ""),
                    "markdown": data.get("markdown", ""),
                })
            except Exception as e:
                logger.warning(f"Failed to load parsed doc {doc_path}: {e}")
    return docs


def _build_chunk_map(chunks: list[Chunk]) -> dict[str, list[str]]:
    """Build a mapping from source_doc_id -> list of chunk_ids."""
    mapping: dict[str, list[str]] = {}
    for chunk in chunks:
        doc_id = chunk.metadata.source_doc_id
        if doc_id:
            mapping.setdefault(doc_id, []).append(chunk.metadata.chunk_id)
    return mapping


def _guess_doc_type(file_type: str, markdown: str) -> str:
    """Guess document type from file extension and content."""
    if file_type in ("pdf",):
        if any(term in markdown[:500].lower() for term in ["פוליסה", "תנאי", "policy"]):
            return "policy"
        return "document"
    if "שאלות" in markdown[:300] or "faq" in markdown[:300].lower():
        return "faq"
    return "webpage"


def _load_checkpoint(checkpoint_path: Path) -> set[str]:
    """Load set of already-processed doc_ids."""
    if not checkpoint_path.exists():
        return set()
    try:
        data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        return set(data.get("completed_doc_ids", []))
    except Exception:
        return set()


def _save_checkpoint(checkpoint_path: Path, completed: set[str]) -> None:
    """Save checkpoint with completed doc_ids."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    data = {"completed_doc_ids": sorted(completed)}
    checkpoint_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _parse_llm_response(response: str) -> dict:
    """Parse LLM JSON response, stripping markdown fences."""
    response = response.strip()
    response = re.sub(r"^```(?:json)?\s*", "", response)
    response = re.sub(r"\s*```$", "", response)
    return json.loads(response)


def build_document_cards(
    parsed_dir: Path = Path("data/parsed"),
    chunks: list[Chunk] | None = None,
    output_dir: Path = Path("data/hierarchy"),
    claude_client: ClaudeClient | None = None,
    gemini_client: GeminiClient | None = None,
    ollama_client: OllamaClient | None = None,
    llm_mode: str = "auto",
    max_content_chars: int = 12000,
) -> list[DocumentCard]:
    """Build document cards for all parsed documents.

    Args:
        parsed_dir: Directory containing parsed/{domain}/{doc_id}.json files.
        chunks: Optional list of all chunks (for building chunk_ids mapping).
        output_dir: Root hierarchy output directory.
        claude_client: Pre-created Claude client (created if None and needed).
        gemini_client: Pre-created Gemini client (created if None and needed).
        ollama_client: Pre-created Ollama client (created if None and needed).
        llm_mode: "claude", "gemini", "ollama", or "auto".
        max_content_chars: Max characters of document markdown to include in prompt.

    Returns:
        List of DocumentCard objects.
    """
    # Create clients as needed
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
                logger.warning("Ollama not available")
                ollama_client = None
        except Exception as e:
            logger.warning(f"Failed to create Ollama client: {e}")

    if not claude_client and not gemini_client and not ollama_client:
        logger.error("No LLM client available. Cannot build document cards.")
        return []

    # Load parsed docs
    parsed_docs = _load_parsed_documents(parsed_dir)
    logger.info(f"Found {len(parsed_docs)} parsed documents")

    # Build chunk mapping
    chunk_map: dict[str, list[str]] = {}
    if chunks:
        chunk_map = _build_chunk_map(chunks)

    # Checkpoint
    checkpoint_path = output_dir / "document_cards_checkpoint.json"
    completed = _load_checkpoint(checkpoint_path)
    if completed:
        logger.info(f"Resuming: {len(completed)} document cards already built")

    cards: list[DocumentCard] = []
    processed = 0
    skipped = 0

    for doc in parsed_docs:
        doc_id = doc["doc_id"]

        # Load existing card if checkpoint says it's done
        if doc_id in completed:
            card_path = output_dir / "documents" / doc["domain"] / f"{doc_id}.json"
            if card_path.exists():
                try:
                    card = DocumentCard.model_validate_json(
                        card_path.read_text("utf-8")
                    )
                    cards.append(card)
                except Exception:
                    completed.discard(doc_id)  # Retry if file is corrupted
                else:
                    skipped += 1
                    continue
            else:
                completed.discard(doc_id)

        processed += 1
        if processed % 20 == 0:
            logger.info(f"  Document card {processed}/{len(parsed_docs) - skipped}...")

        doc_type = _guess_doc_type(doc.get("file_type", ""), doc["markdown"])
        content = doc["markdown"][:max_content_chars]
        chunk_ids = chunk_map.get(doc_id, [])

        prompt = _DOCUMENT_CARD_PROMPT.format(
            title=doc["title"],
            domain=doc["domain"],
            source_url=doc["source_url"],
            doc_type=doc_type,
            content=content,
        )

        result = None

        # Try Claude first
        if claude_client and llm_mode in ("claude", "auto"):
            try:
                response = claude_client.generate(
                    prompt, temperature=0.0, max_tokens=2048
                )
                result = _parse_llm_response(response)
            except Exception as e:
                logger.warning(f"Claude failed for doc {doc_id}: {e}")

        # Try Gemini
        if result is None and gemini_client and llm_mode in ("gemini", "auto"):
            try:
                response = gemini_client.generate(
                    prompt, temperature=0.0, max_tokens=8192,
                    response_mime_type="application/json",
                )
                result = _parse_llm_response(response)
            except Exception as e:
                logger.warning(f"Gemini failed for doc {doc_id}: {e}")

        # Fallback to Ollama
        if result is None and ollama_client and llm_mode in ("ollama", "auto"):
            try:
                logger.info(f"  Falling back to Ollama for doc {doc_id}...")
                response = ollama_client.generate(
                    prompt, temperature=0.0, max_tokens=2048
                )
                result = _parse_llm_response(response)
            except Exception as e:
                logger.warning(f"Ollama failed for doc {doc_id}: {e}")

        if result is None:
            result = {
                "summary": "",
                "key_topics": [],
                "key_facts": [],
                "document_type_note": "",
            }

        card = DocumentCard(
            doc_id=doc_id,
            title=doc["title"],
            domain=doc["domain"],
            source_url=doc["source_url"],
            doc_type=doc_type,
            language="he",
            summary=result.get("summary", ""),
            key_topics=result.get("key_topics", []),
            key_facts=result.get("key_facts", []),
            document_type_note=result.get("document_type_note", ""),
            chunk_count=len(chunk_ids),
            chunk_ids=chunk_ids,
        )
        cards.append(card)

        # Save to disk
        card_dir = output_dir / "documents" / doc["domain"]
        card_dir.mkdir(parents=True, exist_ok=True)
        (card_dir / f"{doc_id}.json").write_text(
            card.model_dump_json(indent=2), encoding="utf-8"
        )

        # Update checkpoint
        completed.add(doc_id)
        _save_checkpoint(checkpoint_path, completed)

    logger.info(
        f"Built {len(cards)} document cards "
        f"({skipped} from checkpoint, {processed} new)"
    )
    return cards
