"""Level 1 builder: generate domain shelf summaries from document summaries.

Groups document summaries by domain and asks Claude to produce
a domain-level overview + document catalog.
"""

import json
import re
from pathlib import Path

from loguru import logger

from config.domains import DOMAINS
from data_pipeline.hierarchy.hierarchy_models import (
    CatalogEntry,
    DocumentSummary,
    DomainSummary,
)
from llm.claude_client import ClaudeClient

_DOMAIN_PROMPT = """You are building a domain-level summary for a section of an insurance company knowledge base.

Domain: {domain} ({domain_he})
Number of documents: {doc_count}

Document summaries:
---
{documents_text}
---

Generate a JSON object with:
1. "overview": A 3-5 sentence overview of this insurance domain — what types of coverage it includes, who needs it, what kinds of questions and topics are covered across all documents.
2. "common_topics": A list of 5-15 common question types or topics that customers ask about in this domain (in both Hebrew and English where appropriate).

Respond with ONLY the JSON object, no markdown fences."""


def _group_docs_by_domain(
    documents: list[DocumentSummary],
) -> dict[str, list[DocumentSummary]]:
    """Group document summaries by domain."""
    groups: dict[str, list[DocumentSummary]] = {}
    for d in documents:
        groups.setdefault(d.domain, []).append(d)
    return groups


def summarize_domains(
    documents: list[DocumentSummary],
    output_dir: Path,
    client: ClaudeClient | None = None,
) -> list[DomainSummary]:
    """Build Level 1 domain summaries from document summaries.

    Args:
        documents: All document summaries from Level 2.
        output_dir: Where to write domain JSON files.
        client: Claude client (created if None).

    Returns:
        List of DomainSummary objects.
    """
    if client is None:
        client = ClaudeClient()

    groups = _group_docs_by_domain(documents)
    logger.info(f"Building domain summaries for {len(groups)} domains...")

    summaries: list[DomainSummary] = []

    for domain_name, domain_docs in groups.items():
        domain_meta = DOMAINS.get(domain_name)
        domain_he = domain_meta.name_he if domain_meta else domain_name

        # Build documents text for prompt
        doc_parts = []
        for d in domain_docs:
            parts = [f"Document: {d.title} (type: {d.doc_type})"]
            if d.summary:
                parts.append(f"  Summary: {d.summary}")
            if d.key_topics:
                parts.append(f"  Topics: {', '.join(d.key_topics[:8])}")
            parts.append(f"  Sections: {d.total_sections}, Chunks: {d.total_chunks}")
            doc_parts.append("\n".join(parts))
        documents_text = "\n\n".join(doc_parts)[:8000]

        prompt = _DOMAIN_PROMPT.format(
            domain=domain_name,
            domain_he=domain_he,
            doc_count=len(domain_docs),
            documents_text=documents_text,
        )

        try:
            response = client.generate(prompt, temperature=0.0, max_tokens=1024)
            response = response.strip()
            response = re.sub(r"^```(?:json)?\s*", "", response)
            response = re.sub(r"\s*```$", "", response)
            result = json.loads(response)
        except Exception as e:
            logger.warning(f"Domain summarization failed for {domain_name}: {e}")
            result = {"overview": "", "common_topics": []}

        catalog = [
            CatalogEntry(
                doc_id=d.doc_id,
                title=d.title,
                doc_type=d.doc_type,
                summary=d.summary,
                key_topics=d.key_topics,
            )
            for d in domain_docs
        ]

        total_chunks = sum(d.total_chunks for d in domain_docs)

        domain_summary = DomainSummary(
            domain=domain_name,
            domain_he=domain_he,
            overview=result.get("overview", ""),
            document_catalog=catalog,
            common_topics=result.get("common_topics", []),
            total_documents=len(domain_docs),
            total_chunks=total_chunks,
        )
        summaries.append(domain_summary)

        # Write to disk
        domain_dir = output_dir / "domains"
        domain_dir.mkdir(parents=True, exist_ok=True)
        (domain_dir / f"{domain_name}.json").write_text(
            domain_summary.model_dump_json(indent=2), encoding="utf-8"
        )

    logger.info(f"Built {len(summaries)} domain summaries")
    return summaries
