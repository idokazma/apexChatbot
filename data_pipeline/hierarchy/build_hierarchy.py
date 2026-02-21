"""Pipeline orchestrator: builds the 3-level hierarchy from parsed documents.

New 3-level flow (document-first):
  1. Document cards (Level 2) — from parsed markdown
  2. Domain shelves (Level 1) — from document cards
  3. Library catalog (Level 0) — from domain shelves
  + Flat chunk index for navigator lookup

Usage:
    python -m data_pipeline.hierarchy.build_hierarchy
    python -m data_pipeline.hierarchy.build_hierarchy --llm auto
    python -m data_pipeline.hierarchy.build_hierarchy --chunks data/chunks/all_chunks.json
"""

import json
from pathlib import Path

from loguru import logger

from data_pipeline.chunker.chunk_models import Chunk
from data_pipeline.hierarchy.catalog_builder import build_catalog
from data_pipeline.hierarchy.document_card_builder import build_document_cards
from data_pipeline.hierarchy.domain_shelf_builder import build_domain_shelves
from data_pipeline.hierarchy.hierarchy_models import LibraryCatalog
from llm.claude_client import ClaudeClient
from llm.gemini_client import GeminiClient
from llm.ollama_client import OllamaClient


def build_hierarchy(
    chunks: list[Chunk] | None = None,
    output_dir: Path = Path("data/hierarchy"),
    parsed_dir: Path = Path("data/parsed"),
    llm_mode: str = "auto",
) -> LibraryCatalog:
    """Build the full 3-level hierarchy.

    Steps:
        1. Document cards (Level 2) from parsed docs
        2. Domain shelves (Level 1) from document cards
        3. Library catalog (Level 0) from domain shelves
        4. Chunk index (flat lookup table)

    Args:
        chunks: Optional list of all chunks (for chunk_ids mapping and index).
            If None, tries to load from data/chunks/all_chunks.json.
        output_dir: Root directory for hierarchy JSON output.
        parsed_dir: Directory with parsed/{domain}/{doc_id}.json files.
        llm_mode: "claude", "ollama", or "auto" (Claude first, Ollama fallback).

    Returns:
        The top-level LibraryCatalog.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load chunks if not provided (needed for chunk_ids and chunk_index)
    if chunks is None:
        chunks_path = Path("data/chunks/all_chunks.json")
        if chunks_path.exists():
            chunks = load_chunks_from_file(chunks_path)
            logger.info(f"Loaded {len(chunks)} chunks from {chunks_path}")
        else:
            logger.warning("No chunks file found — chunk_ids will be empty")
            chunks = []

    # Create LLM clients
    claude_client = None
    gemini_client = None
    ollama_client = None

    if llm_mode in ("claude", "auto"):
        try:
            claude_client = ClaudeClient()
        except Exception as e:
            logger.warning(f"Failed to create Claude client: {e}")

    if llm_mode in ("gemini", "auto"):
        try:
            gemini_client = GeminiClient()
        except Exception as e:
            logger.warning(f"Failed to create Gemini client: {e}")

    if llm_mode in ("ollama", "auto"):
        try:
            ollama_client = OllamaClient()
            if not ollama_client.is_available():
                logger.warning("Ollama not available")
                ollama_client = None
        except Exception as e:
            logger.warning(f"Failed to create Ollama client: {e}")

    logger.info(
        f"Building hierarchy from {parsed_dir} → {output_dir} (mode={llm_mode})"
    )

    # Level 2: Document cards
    logger.info("── Level 2: Document cards ──")
    cards = build_document_cards(
        parsed_dir=parsed_dir,
        chunks=chunks,
        output_dir=output_dir,
        claude_client=claude_client,
        gemini_client=gemini_client,
        ollama_client=ollama_client,
        llm_mode=llm_mode,
    )

    # Level 1: Domain shelves
    logger.info("── Level 1: Domain shelves ──")
    shelves = build_domain_shelves(
        cards=cards,
        output_dir=output_dir,
        claude_client=claude_client,
        gemini_client=gemini_client,
        ollama_client=ollama_client,
        llm_mode=llm_mode,
    )

    # Level 0: Catalog
    logger.info("── Level 0: Library catalog ──")
    catalog = build_catalog(
        shelves, output_dir,
        claude_client=claude_client,
        gemini_client=gemini_client,
    )

    # Chunk index
    if chunks:
        _save_chunk_index(chunks, output_dir)

    logger.info(
        f"Hierarchy complete: {len(cards)} document cards, "
        f"{len(shelves)} domain shelves, catalog built"
    )
    return catalog


def _save_chunk_index(chunks: list[Chunk], output_dir: Path) -> None:
    """Save a flat JSON index mapping chunk_id -> chunk data."""
    index: dict[str, dict] = {}
    for chunk in chunks:
        index[chunk.metadata.chunk_id] = {
            "content": chunk.content,
            "content_with_context": chunk.content_with_context,
            "source_url": chunk.metadata.source_url,
            "source_doc_title": chunk.metadata.source_doc_title,
            "source_doc_id": chunk.metadata.source_doc_id,
            "domain": chunk.metadata.domain,
            "section_path": chunk.metadata.section_path,
            "page_number": chunk.metadata.page_number,
            "source_file_path": chunk.metadata.source_file_path,
            "language": chunk.metadata.language,
            "doc_type": chunk.metadata.doc_type,
            "chunk_index": chunk.metadata.chunk_index,
        }

    out_path = output_dir / "chunk_index.json"
    out_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"Saved chunk index: {len(index)} chunks → {out_path}")


def load_chunks_from_file(chunks_path: Path) -> list[Chunk]:
    """Load chunks from the pipeline's all_chunks.json."""
    raw = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks = []
    for item in raw:
        if isinstance(item, dict):
            chunks.append(Chunk(**item))
    return chunks


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build 3-level hierarchy")
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("data/chunks/all_chunks.json"),
        help="Path to all_chunks.json (optional)",
    )
    parser.add_argument(
        "--parsed",
        type=Path,
        default=Path("data/parsed"),
        help="Path to parsed documents directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/hierarchy"),
        help="Output directory for hierarchy",
    )
    parser.add_argument(
        "--llm",
        choices=["claude", "gemini", "ollama", "auto"],
        default="auto",
        help="LLM mode: claude, gemini, ollama, or auto (default: auto)",
    )
    args = parser.parse_args()

    chunks = None
    if args.chunks.exists():
        chunks = load_chunks_from_file(args.chunks)

    build_hierarchy(
        chunks=chunks,
        output_dir=args.output,
        parsed_dir=args.parsed,
        llm_mode=args.llm,
    )
