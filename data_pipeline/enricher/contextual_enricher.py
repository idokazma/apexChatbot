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

BATCH_SIZE = 10

_BATCH_ENRICHMENT_PROMPT = """You are processing chunks from Israeli insurance documents for a RAG system.

Below are {count} chunks. For EACH chunk, generate a JSON object with:
1. "summary": A 1-2 sentence summary of the key information (in the same language as the content)
2. "keywords": A list of 5-15 important keywords/phrases for search (include both Hebrew and English terms where applicable)
3. "key_facts": A list of specific facts mentioned (amounts, dates, conditions, exclusions) - empty list if none

Respond with a JSON array of {count} objects, one per chunk, in the same order.
Respond with ONLY the JSON array, no markdown fences or extra text.

{chunks_text}"""


def _format_chunk_for_prompt(idx: int, chunk: Chunk) -> str:
    """Format a single chunk for inclusion in the batch prompt."""
    return (
        f"--- Chunk {idx + 1} ---\n"
        f"Document: {chunk.metadata.source_doc_title or 'Unknown'}\n"
        f"Domain: {chunk.metadata.domain or 'Unknown'}\n"
        f"Section: {chunk.metadata.section_path or ''}\n"
        f"Text:\n{chunk.content[:2000]}\n"
    )


def enrich_batch(
    chunks: list[Chunk],
    client: ClaudeClient,
) -> list[dict]:
    """Enrich a batch of chunks in a single API call.

    Args:
        chunks: Batch of chunks to enrich.
        client: Claude API client.

    Returns:
        List of dicts with 'summary', 'keywords', 'key_facts' per chunk.
    """
    chunks_text = "\n".join(
        _format_chunk_for_prompt(i, c) for i, c in enumerate(chunks)
    )

    prompt = _BATCH_ENRICHMENT_PROMPT.format(
        count=len(chunks),
        chunks_text=chunks_text,
    )

    empty_result = {"summary": "", "keywords": [], "key_facts": []}

    try:
        response = client.generate(prompt, temperature=0.0, max_tokens=4096)
        response = response.strip()
        response = re.sub(r"^```(?:json)?\s*", "", response)
        response = re.sub(r"\s*```$", "", response)
        results = json.loads(response)

        if not isinstance(results, list):
            logger.warning("Batch enrichment returned non-list, falling back")
            return [empty_result] * len(chunks)

        # Pad or trim to match chunk count
        while len(results) < len(chunks):
            results.append(empty_result)

        return [
            {
                "summary": r.get("summary", ""),
                "keywords": r.get("keywords", []),
                "key_facts": r.get("key_facts", []),
            }
            for r in results[: len(chunks)]
        ]
    except (json.JSONDecodeError, Exception) as e:
        chunk_ids = [c.metadata.chunk_id for c in chunks]
        logger.warning(f"Batch enrichment failed for {len(chunks)} chunks ({chunk_ids[0]}...): {e}")
        return [empty_result] * len(chunks)


def _apply_enrichment(chunk: Chunk, enrichment: dict) -> bool:
    """Apply enrichment data to a chunk. Returns True if successful."""
    if not enrichment["summary"]:
        return False

    chunk.metadata.keywords = enrichment["keywords"]
    chunk.metadata.summary = enrichment["summary"]
    chunk.metadata.key_facts = enrichment["key_facts"]

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
    return True


def enrich_chunks(
    chunks: list[Chunk],
    batch_size: int = BATCH_SIZE,
    batch_delay: float = 0.1,
) -> list[Chunk]:
    """Enrich all chunks with LLM-generated summaries and keywords.

    Processes chunks in batches for speed (batch_size chunks per API call).

    Args:
        chunks: List of chunks to enrich.
        batch_size: Number of chunks per API call.
        batch_delay: Delay between API calls to avoid rate limiting.

    Returns:
        The enriched chunks (same list, modified in-place).
    """
    client = ClaudeClient()
    total = len(chunks)
    enriched_count = 0
    failed_count = 0

    logger.info(
        f"Enriching {total} chunks in batches of {batch_size} "
        f"(~{total // batch_size + 1} API calls)..."
    )

    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = total // batch_size + 1

        logger.info(f"  Batch {batch_num}/{total_batches} (chunk {i}/{total})...")

        results = enrich_batch(batch, client)

        for chunk, enrichment in zip(batch, results):
            if _apply_enrichment(chunk, enrichment):
                enriched_count += 1
            else:
                failed_count += 1

        time.sleep(batch_delay)

    logger.info(
        f"Enrichment complete: {enriched_count} enriched, {failed_count} failed "
        f"out of {total} chunks"
    )
    return chunks
