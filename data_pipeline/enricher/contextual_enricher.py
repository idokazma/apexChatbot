"""LLM-powered contextual enrichment for chunks.

For each chunk, generates:
- A short summary capturing the key information
- Keywords for BM25/TF-IDF indexing
- Key facts (coverage amounts, conditions, exclusions)

This is the single highest-impact preprocessing step: it makes chunks
"pre-understood" so both keyword search and semantic search work better.
"""

import json
import re
import time

from loguru import logger

from data_pipeline.chunker.chunk_models import Chunk
from llm.claude_client import ClaudeClient

_ENRICHMENT_PROMPT = """You are processing a chunk from an Israeli insurance document for a RAG system.

Document: {doc_title}
Domain: {domain}
Section: {section_path}
Previous chunk context: {prev_context}

Current chunk text:
---
{content}
---

Generate a JSON object with:
1. "summary": A 1-2 sentence summary of the key information in this chunk (in the same language as the content)
2. "keywords": A list of 5-15 important keywords/phrases for search (include both Hebrew and English terms where applicable)
3. "key_facts": A list of specific facts mentioned (amounts, dates, conditions, exclusions) - empty list if none

Respond with ONLY the JSON object, no markdown fences."""


def enrich_chunk(
    chunk: Chunk,
    prev_chunk: Chunk | None,
    client: ClaudeClient,
) -> dict:
    """Enrich a single chunk with LLM-generated context.

    Args:
        chunk: The chunk to enrich.
        prev_chunk: The previous chunk (for context continuity).
        client: Claude API client.

    Returns:
        Dict with 'summary', 'keywords', 'key_facts'.
    """
    prev_context = ""
    if prev_chunk:
        # Use previous chunk's summary if already enriched, else first 200 chars
        prev_context = prev_chunk.content[:200]

    prompt = _ENRICHMENT_PROMPT.format(
        doc_title=chunk.metadata.source_doc_title or "Unknown",
        domain=chunk.metadata.domain or "Unknown",
        section_path=chunk.metadata.section_path or "",
        prev_context=prev_context or "(first chunk in document)",
        content=chunk.content[:2000],
    )

    try:
        response = client.generate(prompt, temperature=0.0, max_tokens=512)
        # Parse JSON from response (handle potential markdown fences)
        response = response.strip()
        response = re.sub(r"^```(?:json)?\s*", "", response)
        response = re.sub(r"\s*```$", "", response)
        result = json.loads(response)
        return {
            "summary": result.get("summary", ""),
            "keywords": result.get("keywords", []),
            "key_facts": result.get("key_facts", []),
        }
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Enrichment failed for chunk {chunk.metadata.chunk_id}: {e}")
        return {"summary": "", "keywords": [], "key_facts": []}


def enrich_chunks(
    chunks: list[Chunk],
    batch_delay: float = 0.1,
) -> list[Chunk]:
    """Enrich all chunks with LLM-generated summaries and keywords.

    Modifies chunks in-place: updates content_with_context and adds
    enrichment data to metadata.

    Args:
        chunks: List of chunks to enrich.
        batch_delay: Delay between API calls to avoid rate limiting.

    Returns:
        The enriched chunks (same list, modified in-place).
    """
    client = ClaudeClient()
    total = len(chunks)
    enriched_count = 0
    failed_count = 0

    logger.info(f"Enriching {total} chunks with contextual summaries and keywords...")

    # Group chunks by document for sequential processing
    doc_chunks: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        doc_key = chunk.metadata.source_url or chunk.metadata.source_doc_title
        doc_chunks.setdefault(doc_key, []).append(chunk)

    # Sort each document's chunks by index
    for doc_key in doc_chunks:
        doc_chunks[doc_key].sort(key=lambda c: c.metadata.chunk_index)

    processed = 0
    for doc_key, doc_chunk_list in doc_chunks.items():
        prev_chunk = None
        for chunk in doc_chunk_list:
            processed += 1
            if processed % 50 == 0:
                logger.info(f"  Enriching chunk {processed}/{total}...")

            enrichment = enrich_chunk(chunk, prev_chunk, client)

            if enrichment["summary"]:
                # Store enrichment in metadata
                chunk.metadata.keywords = enrichment["keywords"]
                chunk.metadata.summary = enrichment["summary"]
                chunk.metadata.key_facts = enrichment["key_facts"]

                # Rebuild content_with_context with enrichment
                header_parts = []
                if chunk.metadata.source_doc_title:
                    header_parts.append(f"Document: {chunk.metadata.source_doc_title}")
                if chunk.metadata.domain:
                    header_parts.append(f"Domain: {chunk.metadata.domain}")
                if chunk.metadata.section_path:
                    header_parts.append(f"Section: {chunk.metadata.section_path}")
                header = " | ".join(header_parts)

                keywords_str = ", ".join(enrichment["keywords"][:10])
                chunk.content_with_context = (
                    f"[{header}]\n"
                    f"Summary: {enrichment['summary']}\n"
                    f"Keywords: {keywords_str}\n\n"
                    f"{chunk.content}"
                )
                enriched_count += 1
            else:
                failed_count += 1

            prev_chunk = chunk
            time.sleep(batch_delay)

    logger.info(
        f"Enrichment complete: {enriched_count} enriched, {failed_count} failed "
        f"out of {total} chunks"
    )
    return chunks
