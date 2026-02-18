"""File-based hierarchy store: loads and caches pre-built JSON summaries.

All hierarchy data lives as JSON files on disk. This store provides
typed accessors that load files on first access and cache them in memory.
"""

import json
from pathlib import Path

from loguru import logger

from data_pipeline.hierarchy.hierarchy_models import (
    DocumentSummary,
    DomainSummary,
    LibraryCatalog,
    SectionSummary,
)


class HierarchyStore:
    """Loads the pre-built hierarchy from disk with in-memory caching."""

    def __init__(self, hierarchy_dir: Path = Path("data/hierarchy")):
        self.hierarchy_dir = hierarchy_dir
        self._catalog: LibraryCatalog | None = None
        self._domains: dict[str, DomainSummary] = {}
        self._documents: dict[str, DocumentSummary] = {}  # keyed by doc_id
        self._sections: dict[str, SectionSummary] = {}  # keyed by section_id
        self._chunk_index: dict[str, dict] | None = None

    # ── Level 0 ────────────────────────────────────────────────────

    def load_catalog(self) -> LibraryCatalog:
        """Load the top-level library catalog."""
        if self._catalog is not None:
            return self._catalog

        path = self.hierarchy_dir / "catalog.json"
        if not path.exists():
            raise FileNotFoundError(f"Catalog not found: {path}")

        self._catalog = LibraryCatalog.model_validate_json(path.read_text("utf-8"))
        logger.debug(f"Loaded catalog: {self._catalog.total_domains} domains")
        return self._catalog

    # ── Level 1 ────────────────────────────────────────────────────

    def load_domain(self, domain: str) -> DomainSummary:
        """Load a domain shelf summary."""
        if domain in self._domains:
            return self._domains[domain]

        path = self.hierarchy_dir / "domains" / f"{domain}.json"
        if not path.exists():
            raise FileNotFoundError(f"Domain not found: {path}")

        ds = DomainSummary.model_validate_json(path.read_text("utf-8"))
        self._domains[domain] = ds
        logger.debug(f"Loaded domain '{domain}': {ds.total_documents} documents")
        return ds

    # ── Level 2 ────────────────────────────────────────────────────

    def load_document(self, domain: str, doc_id: str) -> DocumentSummary:
        """Load a document TOC."""
        cache_key = f"{domain}/{doc_id}"
        if cache_key in self._documents:
            return self._documents[cache_key]

        path = self.hierarchy_dir / "documents" / domain / f"{doc_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        doc = DocumentSummary.model_validate_json(path.read_text("utf-8"))
        self._documents[cache_key] = doc
        logger.debug(f"Loaded document '{doc.title}': {doc.total_sections} sections")
        return doc

    # ── Level 3 ────────────────────────────────────────────────────

    def load_section(self, domain: str, doc_id: str, section_id: str) -> SectionSummary:
        """Load a section detail summary."""
        if section_id in self._sections:
            return self._sections[section_id]

        path = self.hierarchy_dir / "sections" / domain / doc_id / f"{section_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Section not found: {path}")

        sec = SectionSummary.model_validate_json(path.read_text("utf-8"))
        self._sections[section_id] = sec
        logger.debug(f"Loaded section '{sec.section_path}': {sec.chunk_count} chunks")
        return sec

    # ── Level 4: Raw chunks ────────────────────────────────────────

    def load_chunk_index(self) -> dict[str, dict]:
        """Load the flat chunk index (chunk_id -> chunk data)."""
        if self._chunk_index is not None:
            return self._chunk_index

        path = self.hierarchy_dir / "chunk_index.json"
        if not path.exists():
            raise FileNotFoundError(f"Chunk index not found: {path}")

        self._chunk_index = json.loads(path.read_text("utf-8"))
        logger.debug(f"Loaded chunk index: {len(self._chunk_index)} chunks")
        return self._chunk_index

    def load_chunks(self, chunk_ids: list[str]) -> list[dict]:
        """Load specific chunks by their IDs.

        Returns list of chunk dicts in the same format as the existing
        retriever output, so the generator node can consume them directly.
        """
        index = self.load_chunk_index()
        results = []
        for cid in chunk_ids:
            chunk_data = index.get(cid)
            if chunk_data:
                results.append({
                    "chunk_id": cid,
                    "content": chunk_data.get("content", ""),
                    "content_with_context": chunk_data.get("content_with_context", ""),
                    "source_url": chunk_data.get("source_url", ""),
                    "source_doc_title": chunk_data.get("source_doc_title", ""),
                    "source_doc_id": chunk_data.get("source_doc_id", ""),
                    "domain": chunk_data.get("domain", ""),
                    "section_path": chunk_data.get("section_path", ""),
                    "page_number": chunk_data.get("page_number"),
                    "source_file_path": chunk_data.get("source_file_path", ""),
                    "language": chunk_data.get("language", "he"),
                    "doc_type": chunk_data.get("doc_type", ""),
                    "chunk_index": chunk_data.get("chunk_index", 0),
                })
            else:
                logger.warning(f"Chunk not found in index: {cid}")
        return results

    def is_ready(self) -> bool:
        """Check if the hierarchy data exists on disk."""
        return (self.hierarchy_dir / "catalog.json").exists()
