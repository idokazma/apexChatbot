"""Pipeline orchestrator: builds the full 4-level hierarchy from chunks.

Usage:
    from data_pipeline.hierarchy.build_hierarchy import build_hierarchy
    build_hierarchy(chunks, output_dir=Path("data/hierarchy"))

Or via CLI:
    python -m data_pipeline.hierarchy.build_hierarchy --chunks data/chunks/all_chunks.json
"""

import json
from pathlib import Path

from loguru import logger

from data_pipeline.chunker.chunk_models import Chunk
from data_pipeline.hierarchy.catalog_builder import build_catalog
from data_pipeline.hierarchy.document_summarizer import summarize_documents
from data_pipeline.hierarchy.domain_summarizer import summarize_domains
from data_pipeline.hierarchy.hierarchy_models import LibraryCatalog
from data_pipeline.hierarchy.section_summarizer import summarize_sections
from llm.claude_client import ClaudeClient


def build_hierarchy(
    chunks: list[Chunk],
    output_dir: Path = Path("data/hierarchy"),
    client: ClaudeClient | None = None,
) -> LibraryCatalog:
    """Build the full 4-level hierarchy from raw chunks.

    Steps:
        1. Section summaries (Level 3) from chunks
        2. Document TOCs (Level 2) from sections
        3. Domain shelves (Level 1) from documents
        4. Library catalog (Level 0) from domains

    Args:
        chunks: All chunks from the chunking pipeline.
        output_dir: Root directory for hierarchy JSON output.
        client: Claude client (shared across all levels).

    Returns:
        The top-level LibraryCatalog.
    """
    if client is None:
        client = ClaudeClient()

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Building hierarchy from {len(chunks)} chunks → {output_dir}")

    # Level 3: Sections
    logger.info("── Level 3: Section summaries ──")
    sections = summarize_sections(chunks, output_dir, client)

    # Level 2: Documents
    logger.info("── Level 2: Document summaries ──")
    documents = summarize_documents(sections, output_dir, client)

    # Level 1: Domains
    logger.info("── Level 1: Domain summaries ──")
    domains = summarize_domains(documents, output_dir, client)

    # Level 0: Catalog
    logger.info("── Level 0: Library catalog ──")
    catalog = build_catalog(domains, output_dir, client)

    # Also save a flat chunk index for quick lookup at navigation time
    _save_chunk_index(chunks, output_dir)

    logger.info(
        f"Hierarchy complete: {len(sections)} sections, "
        f"{len(documents)} documents, {len(domains)} domains"
    )
    return catalog


def _save_chunk_index(chunks: list[Chunk], output_dir: Path) -> None:
    """Save a flat JSON index mapping chunk_id -> chunk data.

    This allows the navigator to load specific chunks by ID without
    needing ChromaDB or any vector store.
    """
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
    out_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
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

    parser = argparse.ArgumentParser(description="Build hierarchy from chunks")
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("data/chunks/all_chunks.json"),
        help="Path to all_chunks.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/hierarchy"),
        help="Output directory for hierarchy",
    )
    args = parser.parse_args()

    chunks = load_chunks_from_file(args.chunks)
    build_hierarchy(chunks, args.output)
