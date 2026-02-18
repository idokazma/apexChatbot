"""Level 3 builder: generate section summaries from raw chunks.

Groups chunks by (source_doc_id, section_path) and asks Claude to
summarize each group into a SectionSummary.
"""

import hashlib
import json
import re
from pathlib import Path

from loguru import logger

from data_pipeline.chunker.chunk_models import Chunk
from data_pipeline.hierarchy.hierarchy_models import SectionSummary
from llm.claude_client import ClaudeClient

_SECTION_PROMPT = """You are summarizing a section of an Israeli insurance document for a hierarchical search index.

Document: {doc_title}
Domain: {domain}
Section path: {section_path}
Number of chunks: {chunk_count}

Section content (all chunks concatenated):
---
{content}
---

Generate a JSON object with:
1. "summary": A 2-3 sentence summary of what this section covers. Be specific about topics, conditions, amounts, and exclusions.
2. "topics": A list of 3-10 questions a customer might ask that this section can answer (in the same language as the content).
3. "key_details": A list of specific facts — amounts, percentages, deadlines, conditions, exclusions. Empty list if none.

Respond with ONLY the JSON object, no markdown fences."""


def _section_id(doc_id: str, section_path: str) -> str:
    """Deterministic ID for a section."""
    raw = f"{doc_id}::{section_path}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _group_chunks_by_section(
    chunks: list[Chunk],
) -> dict[tuple[str, str, str], list[Chunk]]:
    """Group chunks by (source_doc_id, section_path, domain).

    Returns dict mapping (doc_id, section_path, domain) -> list of Chunk.
    """
    groups: dict[tuple[str, str, str], list[Chunk]] = {}
    for chunk in chunks:
        key = (
            chunk.metadata.source_doc_id,
            chunk.metadata.section_path or "(root)",
            chunk.metadata.domain,
        )
        groups.setdefault(key, []).append(chunk)

    # Sort chunks within each group by chunk_index
    for key in groups:
        groups[key].sort(key=lambda c: c.metadata.chunk_index)

    return groups


def summarize_sections(
    chunks: list[Chunk],
    output_dir: Path,
    client: ClaudeClient | None = None,
) -> list[SectionSummary]:
    """Build Level 3 section summaries from raw chunks.

    Args:
        chunks: All chunks from the pipeline.
        output_dir: Where to write section JSON files.
        client: Claude client (created if None).

    Returns:
        List of SectionSummary objects.
    """
    if client is None:
        client = ClaudeClient()

    groups = _group_chunks_by_section(chunks)
    logger.info(f"Building section summaries for {len(groups)} sections...")

    summaries: list[SectionSummary] = []
    processed = 0

    for (doc_id, section_path, domain), group in groups.items():
        processed += 1
        if processed % 20 == 0:
            logger.info(f"  Section {processed}/{len(groups)}...")

        doc_title = group[0].metadata.source_doc_title or "Unknown"
        chunk_ids = [c.metadata.chunk_id for c in group]
        content = "\n\n".join(c.content for c in group)

        # Truncate to avoid token limits
        content_truncated = content[:6000]

        prompt = _SECTION_PROMPT.format(
            doc_title=doc_title,
            domain=domain,
            section_path=section_path,
            chunk_count=len(group),
            content=content_truncated,
        )

        sid = _section_id(doc_id, section_path)

        try:
            response = client.generate(prompt, temperature=0.0, max_tokens=1024)
            response = response.strip()
            response = re.sub(r"^```(?:json)?\s*", "", response)
            response = re.sub(r"\s*```$", "", response)
            result = json.loads(response)
        except Exception as e:
            logger.warning(f"Section summarization failed for {sid}: {e}")
            result = {"summary": "", "topics": [], "key_details": []}

        section = SectionSummary(
            section_id=sid,
            source_doc_id=doc_id,
            source_doc_title=doc_title,
            domain=domain,
            section_path=section_path,
            summary=result.get("summary", ""),
            topics=result.get("topics", []),
            key_details=result.get("key_details", []),
            chunk_ids=chunk_ids,
            chunk_count=len(group),
        )
        summaries.append(section)

        # Write to disk
        section_dir = output_dir / "sections" / domain / doc_id
        section_dir.mkdir(parents=True, exist_ok=True)
        (section_dir / f"{sid}.json").write_text(
            section.model_dump_json(indent=2), encoding="utf-8"
        )

    logger.info(f"Built {len(summaries)} section summaries")
    return summaries
