"""Pydantic models for the 4-level document hierarchy.

Level 0: LibraryCatalog — one summary for the entire knowledge base
Level 1: DomainSummary  — one summary per insurance domain (8 total)
Level 2: DocumentSummary — one TOC per source document
Level 3: SectionSummary  — one summary per section within a document
Level 4: Raw chunks      — already exist (Chunk model in chunker/)
"""

from pydantic import BaseModel, Field


# ── Level 3: Section ──────────────────────────────────────────────


class SectionSummary(BaseModel):
    """Summary of a document section (group of chunks under one heading path)."""

    section_id: str  # hash of doc_id + section_path
    source_doc_id: str
    source_doc_title: str = ""
    domain: str = ""
    section_path: str = ""  # "H1 > H2 > H3"
    summary: str = ""  # 2-3 sentences
    topics: list[str] = Field(default_factory=list)  # questions this section answers
    key_details: list[str] = Field(default_factory=list)  # amounts, dates, conditions
    chunk_ids: list[str] = Field(default_factory=list)  # pointers to raw chunks
    chunk_count: int = 0


# ── Level 2: Document ─────────────────────────────────────────────


class TOCEntry(BaseModel):
    """A single row in a document's table of contents."""

    section_id: str  # pointer to Level 3
    section_path: str
    summary: str = ""  # 1-2 sentences
    topics: list[str] = Field(default_factory=list)


class DocumentSummary(BaseModel):
    """Summary and table of contents for a single source document."""

    doc_id: str
    title: str = ""
    domain: str = ""
    source_url: str = ""
    doc_type: str = "webpage"  # "policy_document", "faq", "webpage"
    language: str = "he"
    summary: str = ""  # 3-5 sentence overview
    table_of_contents: list[TOCEntry] = Field(default_factory=list)
    key_topics: list[str] = Field(default_factory=list)
    total_sections: int = 0
    total_chunks: int = 0


# ── Level 1: Domain ───────────────────────────────────────────────


class CatalogEntry(BaseModel):
    """A single document entry within a domain shelf."""

    doc_id: str
    title: str = ""
    doc_type: str = "webpage"
    summary: str = ""  # 1-2 sentences
    key_topics: list[str] = Field(default_factory=list)


class DomainSummary(BaseModel):
    """Summary of all documents in one insurance domain."""

    domain: str
    domain_he: str = ""
    overview: str = ""  # 3-5 sentence overview
    document_catalog: list[CatalogEntry] = Field(default_factory=list)
    common_topics: list[str] = Field(default_factory=list)
    total_documents: int = 0
    total_chunks: int = 0


# ── Level 0: Library Catalog ──────────────────────────────────────


class DomainOverview(BaseModel):
    """Brief overview of one domain within the library catalog."""

    domain: str
    domain_he: str = ""
    summary: str = ""  # 2-3 sentences
    handles_questions_like: list[str] = Field(default_factory=list)
    document_count: int = 0


class LibraryCatalog(BaseModel):
    """Top-level catalog: the entry point for agentic navigation."""

    domains: list[DomainOverview] = Field(default_factory=list)
    total_documents: int = 0
    total_domains: int = 0
    generated_at: str = ""
