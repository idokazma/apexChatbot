"""Level 2 builder: generate document TOCs from section summaries.

Groups section summaries by source_doc_id and asks Claude to produce
a document-level summary + table of contents.
"""

import json
import re
from pathlib import Path

from loguru import logger

from data_pipeline.hierarchy.hierarchy_models import (
    DocumentSummary,
    SectionSummary,
    TOCEntry,
)
from llm.claude_client import ClaudeClient

_DOCUMENT_PROMPT = """You are building a table-of-contents for an Israeli insurance document in a hierarchical search index.

Document title: {title}
Domain: {domain}
Document type: {doc_type}
Number of sections: {section_count}

Section summaries:
---
{sections_text}
---

Generate a JSON object with:
1. "summary": A 3-5 sentence overview of the entire document — what topics it covers, what type of document it is, who it's relevant for.
2. "key_topics": A list of 5-15 questions/topics that this document can answer (in the same language as the content).

Respond with ONLY the JSON object, no markdown fences."""


def _group_sections_by_doc(
    sections: list[SectionSummary],
) -> dict[str, list[SectionSummary]]:
    """Group section summaries by source_doc_id."""
    groups: dict[str, list[SectionSummary]] = {}
    for s in sections:
        groups.setdefault(s.source_doc_id, []).append(s)
    return groups


def summarize_documents(
    sections: list[SectionSummary],
    output_dir: Path,
    client: ClaudeClient | None = None,
) -> list[DocumentSummary]:
    """Build Level 2 document summaries from section summaries.

    Args:
        sections: All section summaries from Level 3.
        output_dir: Where to write document JSON files.
        client: Claude client (created if None).

    Returns:
        List of DocumentSummary objects.
    """
    if client is None:
        client = ClaudeClient()

    groups = _group_sections_by_doc(sections)
    logger.info(f"Building document summaries for {len(groups)} documents...")

    summaries: list[DocumentSummary] = []
    processed = 0

    for doc_id, doc_sections in groups.items():
        processed += 1
        if processed % 10 == 0:
            logger.info(f"  Document {processed}/{len(groups)}...")

        first = doc_sections[0]
        title = first.source_doc_title
        domain = first.domain

        # Guess doc_type from any section metadata (all share the same doc)
        doc_type = "webpage"

        # Build sections text for the prompt
        sections_text_parts = []
        for s in doc_sections:
            parts = [f"Section: {s.section_path}"]
            if s.summary:
                parts.append(f"  Summary: {s.summary}")
            if s.topics:
                parts.append(f"  Topics: {', '.join(s.topics[:5])}")
            if s.key_details:
                parts.append(f"  Key details: {', '.join(s.key_details[:5])}")
            sections_text_parts.append("\n".join(parts))
        sections_text = "\n\n".join(sections_text_parts)

        # Truncate if too long
        sections_text = sections_text[:8000]

        prompt = _DOCUMENT_PROMPT.format(
            title=title,
            domain=domain,
            doc_type=doc_type,
            section_count=len(doc_sections),
            sections_text=sections_text,
        )

        try:
            response = client.generate(prompt, temperature=0.0, max_tokens=1024)
            response = response.strip()
            response = re.sub(r"^```(?:json)?\s*", "", response)
            response = re.sub(r"\s*```$", "", response)
            result = json.loads(response)
        except Exception as e:
            logger.warning(f"Document summarization failed for {doc_id}: {e}")
            result = {"summary": "", "key_topics": []}

        toc = [
            TOCEntry(
                section_id=s.section_id,
                section_path=s.section_path,
                summary=s.summary,
                topics=s.topics,
            )
            for s in doc_sections
        ]

        total_chunks = sum(s.chunk_count for s in doc_sections)

        doc_summary = DocumentSummary(
            doc_id=doc_id,
            title=title,
            domain=domain,
            source_url=first.source_doc_title,  # best we have at section level
            doc_type=doc_type,
            summary=result.get("summary", ""),
            table_of_contents=toc,
            key_topics=result.get("key_topics", []),
            total_sections=len(doc_sections),
            total_chunks=total_chunks,
        )
        summaries.append(doc_summary)

        # Write to disk
        doc_dir = output_dir / "documents" / domain
        doc_dir.mkdir(parents=True, exist_ok=True)
        (doc_dir / f"{doc_id}.json").write_text(
            doc_summary.model_dump_json(indent=2), encoding="utf-8"
        )

    logger.info(f"Built {len(summaries)} document summaries")
    return summaries
