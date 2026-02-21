"""LLM-powered contextual enrichment for chunks.

For each chunk, generates:
- A short summary capturing the key information
- Keywords for BM25/TF-IDF indexing
- Key facts (coverage amounts, conditions, exclusions)

This is the single highest-impact preprocessing step: it makes chunks
"pre-understood" so both keyword search and semantic search work better.

Supports checkpoint/resume and multiple LLM backends (Claude, Ollama, Gemini).
"""

import json
import re
import time
from pathlib import Path

from loguru import logger

from data_pipeline.chunker.chunk_models import Chunk
from llm.claude_client import ClaudeClient
from llm.gemini_client import GeminiClient
from llm.ollama_client import OllamaClient

BATCH_SIZE_CLAUDE = 10
BATCH_SIZE_GEMINI = 5
BATCH_SIZE_OLLAMA = 3
CHUNKS_PATH = Path("data/chunks/all_chunks.json")
INCREMENTAL_SAVE_INTERVAL = 10  # Save all_chunks.json every N batches

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


def _parse_enrichment_response(response: str, count: int) -> list[dict]:
    """Parse LLM response into enrichment dicts, with fallback."""
    empty_result = {"summary": "", "keywords": [], "key_facts": []}

    response = response.strip()
    response = re.sub(r"^```(?:json)?\s*", "", response)
    response = re.sub(r"\s*```$", "", response)

    try:
        results = json.loads(response)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse enrichment JSON: {e}")
        return [empty_result] * count

    # Handle dict-style response (e.g. {"chunk_1": {...}, "chunk_2": {...}})
    if isinstance(results, dict):
        if all(isinstance(v, dict) for v in results.values()):
            results = list(results.values())
        else:
            return [empty_result] * count

    if not isinstance(results, list):
        return [empty_result] * count

    while len(results) < count:
        results.append(empty_result)

    return [
        {
            "summary": r.get("summary", ""),
            "keywords": r.get("keywords", []),
            "key_facts": r.get("key_facts", []),
        }
        for r in results[:count]
    ]


def enrich_batch_with_client(
    chunks: list[Chunk],
    client: ClaudeClient | OllamaClient | GeminiClient,
) -> list[dict]:
    """Enrich a batch of chunks using the given LLM client.

    Args:
        chunks: Batch of chunks to enrich.
        client: Any LLM client with a .generate() method.

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

    kwargs = {"temperature": 0.0, "max_tokens": 8192}
    if isinstance(client, OllamaClient):
        kwargs["format"] = "json"

    response = client.generate(prompt, **kwargs)
    return _parse_enrichment_response(response, len(chunks))


def enrich_batch(
    chunks: list[Chunk],
    client: ClaudeClient,
) -> list[dict]:
    """Enrich a batch using Claude (backward-compatible wrapper)."""
    return enrich_batch_with_client(chunks, client)


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



def _save_chunks_incrementally(chunks: list[Chunk], path: Path) -> None:
    """Save all chunks to disk for crash recovery."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [c.model_dump() for c in chunks]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Incremental save: {len(chunks)} chunks → {path}")


def _create_client(llm_mode: str):
    """Create LLM client based on the mode.

    Returns the client instance or None.
    """
    if llm_mode == "gemini":
        try:
            client = GeminiClient()
            return client
        except Exception as e:
            logger.error(f"Failed to create Gemini client: {e}")
            return None

    if llm_mode == "claude":
        try:
            return ClaudeClient()
        except Exception as e:
            logger.error(f"Failed to create Claude client: {e}")
            return None

    if llm_mode == "ollama":
        try:
            client = OllamaClient()
            if not client.is_available():
                logger.error("Ollama not available")
                return None
            return client
        except Exception as e:
            logger.error(f"Failed to create Ollama client: {e}")
            return None

    # auto mode: try Claude → Gemini → Ollama
    for mode in ("claude", "gemini", "ollama"):
        client = _create_client(mode)
        if client:
            logger.info(f"Auto-selected LLM: {mode}")
            return client

    return None


def _get_batch_size(client, llm_mode: str) -> int:
    """Get optimal batch size for the client type."""
    if isinstance(client, GeminiClient):
        return BATCH_SIZE_GEMINI
    if isinstance(client, ClaudeClient):
        return BATCH_SIZE_CLAUDE
    return BATCH_SIZE_OLLAMA


def enrich_chunks(
    chunks: list[Chunk],
    batch_size: int | None = None,
    batch_delay: float = 0.1,
    llm_mode: str = "auto",
    chunks_save_path: Path = CHUNKS_PATH,
) -> list[Chunk]:
    """Enrich all chunks with LLM-generated summaries and keywords.

    Supports checkpoint/resume and multiple LLM backends.

    Args:
        chunks: List of chunks to enrich.
        batch_size: Number of chunks per API call (auto-selected if None).
        batch_delay: Delay between API calls to avoid rate limiting.
        llm_mode: LLM backend — "claude", "ollama", "gemini", or "auto".
        chunks_save_path: Path for incremental saves of all_chunks.json.

    Returns:
        The enriched chunks (same list, modified in-place).
    """
    client = _create_client(llm_mode)

    if not client:
        logger.error("No LLM client available. Cannot enrich chunks.")
        return chunks

    if batch_size is None:
        batch_size = _get_batch_size(client, llm_mode)

    total = len(chunks)
    enriched_count = 0
    failed_count = 0
    skipped_count = 0

    total_batches = (total + batch_size - 1) // batch_size
    logger.info(
        f"Enriching {total} chunks in batches of {batch_size} "
        f"(~{total_batches} batches, mode={llm_mode})..."
    )

    batches_since_save = 0

    for i in range(0, total, batch_size):
        batch_num = i // batch_size + 1

        batch = chunks[i : i + batch_size]

        # Skip chunks that already have enrichment
        unenriched = [c for c in batch if not c.metadata.summary]
        if not unenriched:
            skipped_count += len(batch)
            continue

        logger.info(
            f"  Batch {batch_num}/{total_batches} "
            f"(chunk {i}/{total}, {len(unenriched)} to enrich)..."
        )

        try:
            results = enrich_batch_with_client(unenriched, client)
        except Exception as e:
            logger.warning(f"Failed batch {batch_num}: {e}")
            empty = {"summary": "", "keywords": [], "key_facts": []}
            results = [empty] * len(unenriched)

        for chunk, enrichment in zip(unenriched, results):
            if _apply_enrichment(chunk, enrichment):
                enriched_count += 1
            else:
                failed_count += 1

        # Incremental save every N batches
        batches_since_save += 1
        if batches_since_save >= INCREMENTAL_SAVE_INTERVAL:
            _save_chunks_incrementally(chunks, chunks_save_path)
            batches_since_save = 0

        time.sleep(batch_delay)

    # Final save
    _save_chunks_incrementally(chunks, chunks_save_path)

    logger.info(
        f"Enrichment complete: {enriched_count} enriched, {failed_count} failed, "
        f"{skipped_count} skipped (checkpoint) out of {total} chunks"
    )
    return chunks
