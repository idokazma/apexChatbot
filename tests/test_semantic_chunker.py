"""Tests for data_pipeline.chunker.semantic_chunker."""

from data_pipeline.chunker.semantic_chunker import (
    SemanticChunker,
    _derive_source_file_path,
    _find_page_number,
    chunk_parsed_documents,
)


class TestDeriveSourceFilePath:
    def test_pdf_url(self):
        result = _derive_source_file_path(
            "apartment",
            "https://media.harel-group.co.il/files/www.harel-group.co.il--2025.pdf",
        )
        assert result == "apartment/files/www.harel-group.co.il--2025.pdf"

    def test_empty_url(self):
        assert _derive_source_file_path("car", "") == ""

    def test_simple_filename(self):
        result = _derive_source_file_path("car", "https://example.com/docs/policy.pdf")
        assert result == "car/files/policy.pdf"

    def test_no_path(self):
        result = _derive_source_file_path("car", "https://example.com")
        assert result == ""


class TestFindPageNumber:
    def test_match_found(self):
        page_map = [
            {"text": "Chapter 1: Introduction to coverage", "page_number": 1},
            {"text": "Chapter 2: Exclusions and limitations", "page_number": 5},
        ]
        result = _find_page_number("Chapter 2: Exclusions and limitations apply...", page_map)
        assert result == 5

    def test_no_match(self):
        page_map = [{"text": "unrelated", "page_number": 1}]
        result = _find_page_number("Different text entirely", page_map)
        assert result is None

    def test_empty_page_map(self):
        assert _find_page_number("any text", []) is None

    def test_none_page_map(self):
        assert _find_page_number("any text", None) is None


class TestSemanticChunker:
    def test_simple_document(self):
        chunker = SemanticChunker(max_chunk_tokens=512, overlap_tokens=0)
        doc = {
            "markdown": "# Title\n\nSome content here.\n\n# Another\n\nMore content.",
            "title": "Test",
            "domain": "car",
            "source_url": "https://example.com/doc",
        }
        chunks = chunker.chunk_document(doc)
        assert len(chunks) >= 2
        assert chunks[0].metadata.domain == "car"
        assert chunks[0].metadata.chunk_index == 0

    def test_empty_markdown(self):
        chunker = SemanticChunker()
        doc = {"markdown": "", "title": "Empty", "domain": "car", "source_url": ""}
        chunks = chunker.chunk_document(doc)
        assert chunks == []

    def test_whitespace_only_markdown(self):
        chunker = SemanticChunker()
        doc = {"markdown": "   \n  \n  ", "title": "WS", "domain": "car", "source_url": ""}
        chunks = chunker.chunk_document(doc)
        assert chunks == []

    def test_no_headers(self):
        chunker = SemanticChunker(max_chunk_tokens=512, overlap_tokens=0)
        doc = {
            "markdown": "Just plain text without any headers.",
            "title": "Plain",
            "domain": "car",
            "source_url": "",
        }
        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 1

    def test_chunk_indices_sequential(self):
        chunker = SemanticChunker(max_chunk_tokens=512, overlap_tokens=0)
        doc = {
            "markdown": "# A\nContent A\n# B\nContent B\n# C\nContent C",
            "title": "Multi",
            "domain": "car",
            "source_url": "",
        }
        chunks = chunker.chunk_document(doc)
        for i, chunk in enumerate(chunks):
            assert chunk.metadata.chunk_index == i
            assert chunk.metadata.total_chunks_in_doc == len(chunks)

    def test_split_by_headers_section_path(self):
        chunker = SemanticChunker(max_chunk_tokens=512, overlap_tokens=0)
        doc = {
            "markdown": "# H1\n\n## H2\n\nContent under H2",
            "title": "Test",
            "domain": "car",
            "source_url": "",
        }
        chunks = chunker.chunk_document(doc)
        # Find chunk with H2 content
        h2_chunks = [c for c in chunks if "H2" in c.metadata.section_path]
        assert len(h2_chunks) >= 1
        assert "H1" in h2_chunks[0].metadata.section_path

    def test_large_section_split(self):
        chunker = SemanticChunker(max_chunk_tokens=20, overlap_tokens=0)
        # Create a document whose single section exceeds max_chunk_tokens
        long_text = " ".join(["word"] * 100)
        doc = {
            "markdown": f"# Section\n{long_text}",
            "title": "Long",
            "domain": "car",
            "source_url": "",
        }
        chunks = chunker.chunk_document(doc)
        assert len(chunks) > 1

    def test_overlap_added(self):
        chunker = SemanticChunker(max_chunk_tokens=20, overlap_tokens=10)
        long_text = " ".join(["word"] * 100)
        doc = {
            "markdown": f"# Section\n{long_text}",
            "title": "Overlap",
            "domain": "car",
            "source_url": "",
        }
        chunks = chunker.chunk_document(doc)
        if len(chunks) > 1:
            # Second chunk should start with overlap marker
            assert chunks[1].content.startswith("...")


class TestChunkParsedDocuments:
    def test_basic(self):
        # Content must exceed min_tokens (default 15) to not be filtered
        long_content = " ".join(["insurance coverage details"] * 10)
        docs = [
            {
                "markdown": f"# Insurance Policy\n{long_content}",
                "title": "Doc 1",
                "domain": "car",
                "source_url": "",
            }
        ]
        chunks = chunk_parsed_documents(docs)
        assert len(chunks) >= 1

    def test_filters_small_chunks(self):
        docs = [
            {
                "markdown": "# A\nhi",  # very short
                "title": "Small",
                "domain": "car",
                "source_url": "",
            }
        ]
        chunks = chunk_parsed_documents(docs, min_tokens=100)
        assert len(chunks) == 0

    def test_empty_docs(self):
        assert chunk_parsed_documents([]) == []
