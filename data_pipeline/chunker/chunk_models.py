"""Pydantic models for document chunks with provenance tracking."""

import uuid

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Metadata attached to every chunk for citation tracking."""

    chunk_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    source_url: str = ""
    source_doc_title: str = ""
    source_doc_id: str = ""  # Hash of source_url for neighbor lookups
    domain: str = ""
    section_path: str = ""  # "H1 > H2 > H3" breadcrumb
    page_number: int | None = None
    language: str = "he"
    doc_type: str = "webpage"
    chunk_index: int = 0
    total_chunks_in_doc: int = 0

    # Contextual enrichment fields (populated by LLM enricher)
    summary: str = ""
    keywords: list[str] = Field(default_factory=list)
    key_facts: list[str] = Field(default_factory=list)


class Chunk(BaseModel):
    """A single chunk of text with full provenance metadata."""

    content: str
    content_with_context: str = ""  # Prefixed with context header
    metadata: ChunkMetadata = Field(default_factory=ChunkMetadata)
    token_count: int = 0

    def model_post_init(self, __context) -> None:
        """Build context-enriched content after initialization."""
        if not self.content_with_context:
            header_parts = []
            if self.metadata.source_doc_title:
                header_parts.append(f"Document: {self.metadata.source_doc_title}")
            if self.metadata.domain:
                header_parts.append(f"Domain: {self.metadata.domain}")
            if self.metadata.section_path:
                header_parts.append(f"Section: {self.metadata.section_path}")
            header = " | ".join(header_parts)
            self.content_with_context = f"[{header}]\n{self.content}" if header else self.content

        if not self.token_count:
            # Rough estimate: ~1.5 tokens per word for Hebrew, ~1.3 for English
            self.token_count = len(self.content.split()) * 2

        # Generate source_doc_id for neighbor chunk lookups
        if not self.metadata.source_doc_id and self.metadata.source_url:
            import hashlib
            self.metadata.source_doc_id = hashlib.md5(
                self.metadata.source_url.encode()
            ).hexdigest()[:12]
