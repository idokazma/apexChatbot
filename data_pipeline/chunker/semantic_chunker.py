"""Intelligent document chunking with structural and semantic splitting."""

import re

from loguru import logger

from config.settings import settings
from data_pipeline.chunker.chunk_models import Chunk, ChunkMetadata


class SemanticChunker:
    """Chunks documents by structure (headers) then enforces size limits."""

    def __init__(
        self,
        max_chunk_tokens: int = settings.max_chunk_tokens,
        overlap_tokens: int = settings.chunk_overlap_tokens,
    ):
        self.max_chunk_tokens = max_chunk_tokens
        self.overlap_tokens = overlap_tokens

    def chunk_document(self, parsed_doc: dict) -> list[Chunk]:
        """Chunk a parsed document into retrieval-ready pieces.

        Args:
            parsed_doc: Dict with 'markdown', 'title', 'domain', 'source_url', etc.

        Returns:
            List of Chunk objects with full provenance metadata.
        """
        markdown = parsed_doc.get("markdown", "")
        if not markdown.strip():
            return []

        # Step 1: Split by headers into structural sections
        sections = self._split_by_headers(markdown)

        # Step 2: For each section, enforce max size
        chunks: list[Chunk] = []
        for section in sections:
            section_chunks = self._enforce_size_limit(
                section["content"], section["section_path"]
            )
            for chunk_text in section_chunks:
                metadata = ChunkMetadata(
                    source_url=parsed_doc.get("source_url", ""),
                    source_doc_title=parsed_doc.get("title", ""),
                    domain=parsed_doc.get("domain", ""),
                    section_path=section["section_path"],
                    language=parsed_doc.get("language", "he"),
                    doc_type=parsed_doc.get("doc_type", "webpage"),
                    page_number=section.get("page_number"),
                )
                chunks.append(Chunk(content=chunk_text, metadata=metadata))

        # Step 3: Set chunk indices
        for i, chunk in enumerate(chunks):
            chunk.metadata.chunk_index = i
            chunk.metadata.total_chunks_in_doc = len(chunks)

        # Step 4: Add overlap between consecutive chunks
        chunks = self._add_overlap(chunks)

        logger.debug(
            f"  Chunked '{parsed_doc.get('title', 'untitled')}' into {len(chunks)} chunks"
        )
        return chunks

    def _split_by_headers(self, markdown: str) -> list[dict]:
        """Split markdown into sections based on header hierarchy."""
        lines = markdown.split("\n")
        sections: list[dict] = []
        current_section_lines: list[str] = []
        heading_stack: list[str] = []  # Track H1 > H2 > H3

        for line in lines:
            header_match = re.match(r"^(#{1,4})\s+(.+)$", line.strip())

            if header_match:
                # Save the previous section
                if current_section_lines:
                    content = "\n".join(current_section_lines).strip()
                    if content:
                        sections.append({
                            "content": content,
                            "section_path": " > ".join(heading_stack) if heading_stack else "",
                        })

                # Update heading stack
                level = len(header_match.group(1))
                heading_text = header_match.group(2).strip()

                # Pop headings at same or deeper level
                while len(heading_stack) >= level:
                    heading_stack.pop()
                heading_stack.append(heading_text)

                current_section_lines = [line]
            else:
                current_section_lines.append(line)

        # Don't forget the last section
        if current_section_lines:
            content = "\n".join(current_section_lines).strip()
            if content:
                sections.append({
                    "content": content,
                    "section_path": " > ".join(heading_stack) if heading_stack else "",
                })

        # If no headers found, treat entire document as one section
        if not sections and markdown.strip():
            sections.append({"content": markdown.strip(), "section_path": ""})

        return sections

    def _enforce_size_limit(self, text: str, section_path: str) -> list[str]:
        """Split text that exceeds max_chunk_tokens into smaller pieces."""
        estimated_tokens = len(text.split()) * 2  # rough estimate for Hebrew
        if estimated_tokens <= self.max_chunk_tokens:
            return [text]

        # Try splitting by paragraphs first
        paragraphs = re.split(r"\n\s*\n", text)
        if len(paragraphs) > 1:
            return self._merge_paragraphs_to_chunks(paragraphs)

        # If single paragraph is too long, split by sentences
        sentences = re.split(r"(?<=[.!?。])\s+", text)
        if len(sentences) > 1:
            return self._merge_paragraphs_to_chunks(sentences)

        # Last resort: split by character count
        words = text.split()
        max_words = self.max_chunk_tokens // 2
        chunks = []
        for i in range(0, len(words), max_words):
            chunk = " ".join(words[i : i + max_words])
            chunks.append(chunk)
        return chunks

    def _merge_paragraphs_to_chunks(self, paragraphs: list[str]) -> list[str]:
        """Merge small paragraphs into chunks that fit within the token limit."""
        chunks: list[str] = []
        current_parts: list[str] = []
        current_tokens = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            para_tokens = len(para.split()) * 2

            if current_tokens + para_tokens > self.max_chunk_tokens and current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = []
                current_tokens = 0

            current_parts.append(para)
            current_tokens += para_tokens

        if current_parts:
            chunks.append("\n\n".join(current_parts))

        return chunks

    def _add_overlap(self, chunks: list[Chunk]) -> list[Chunk]:
        """Add overlap text from neighboring chunks for context continuity."""
        if len(chunks) <= 1 or self.overlap_tokens <= 0:
            return chunks

        for i in range(1, len(chunks)):
            prev_words = chunks[i - 1].content.split()
            overlap_words = self.overlap_tokens // 2  # rough token-to-word
            if len(prev_words) > overlap_words:
                overlap_text = " ".join(prev_words[-overlap_words:])
                chunks[i].content = f"...{overlap_text}\n\n{chunks[i].content}"
                # Rebuild context version
                chunks[i].content_with_context = ""
                chunks[i].model_post_init(None)

        return chunks


def chunk_parsed_documents(parsed_docs: list[dict]) -> list[Chunk]:
    """Chunk a list of parsed documents."""
    chunker = SemanticChunker()
    all_chunks: list[Chunk] = []

    for doc in parsed_docs:
        chunks = chunker.chunk_document(doc)
        all_chunks.extend(chunks)

    logger.info(f"Total chunks created: {len(all_chunks)} from {len(parsed_docs)} documents")
    return all_chunks
