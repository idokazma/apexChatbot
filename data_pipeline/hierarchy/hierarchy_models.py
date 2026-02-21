"""Pydantic models for the 3-level document hierarchy.

Level 0: LibraryCatalog — one summary for the entire knowledge base
Level 1: DomainShelf    — one summary per insurance domain (8 total)
Level 2: DocumentCard   — one rich card per source document (~621 total)
Level 3: Raw chunks     — already exist (Chunk model in chunker/)

The old 4-level hierarchy (sections → documents → domains → catalog) is
replaced by a document-first approach: summarize whole documents, then
group them by domain.
"""

from pydantic import BaseModel, Field


# ── Level 2: Document Card ───────────────────────────────────────


class DocumentCard(BaseModel):
    """Rich summary of a single source document.

    Designed for an LLM to read and decide: "does this document
    contain what I need?"
    """

    doc_id: str
    title: str = ""
    domain: str = ""
    source_url: str = ""
    doc_type: str = "webpage"  # policy, faq, webpage, form
    language: str = "he"
    # -- LLM-generated fields --
    summary: str = ""  # 5-10 sentence overview for an LLM reader
    key_topics: list[str] = Field(default_factory=list)  # 10-20 topics/questions
    key_facts: list[str] = Field(default_factory=list)  # numbers, conditions, exclusions
    document_type_note: str = ""  # "This is a full policy document" etc.
    chunk_count: int = 0
    chunk_ids: list[str] = Field(default_factory=list)


# ── Level 1: Domain Shelf ────────────────────────────────────────


class DocumentCardBrief(BaseModel):
    """Quick reference entry for a document within a domain shelf."""

    doc_id: str
    title: str = ""
    doc_type: str = "webpage"
    summary: str = ""  # 1-2 sentences
    key_topics: list[str] = Field(default_factory=list)


class DocumentGroup(BaseModel):
    """A thematic cluster of related documents within a domain."""

    group_name: str  # e.g., "Policy Terms & Conditions"
    group_summary: str = ""  # What this cluster covers together
    doc_ids: list[str] = Field(default_factory=list)


class DomainShelf(BaseModel):
    """Summary of all documents in one insurance domain."""

    domain: str
    domain_he: str = ""
    overview: str = ""  # What this domain covers overall
    document_groups: list[DocumentGroup] = Field(default_factory=list)
    all_documents: list[DocumentCardBrief] = Field(default_factory=list)
    total_documents: int = 0
    total_chunks: int = 0


# ── Level 0: Library Catalog ─────────────────────────────────────


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


# ── Legacy aliases (backward compatibility during transition) ─────

# These are kept so that existing code that imports them doesn't break.
# They map to the new equivalents.
DomainSummary = DomainShelf
CatalogEntry = DocumentCardBrief
