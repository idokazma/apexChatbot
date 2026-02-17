"""Tests for data_pipeline.chunker.chunk_models."""

from data_pipeline.chunker.chunk_models import Chunk, ChunkMetadata


class TestChunkMetadata:
    def test_default_values(self):
        meta = ChunkMetadata()
        assert len(meta.chunk_id) == 16
        assert meta.source_url == ""
        assert meta.domain == ""
        assert meta.language == "he"
        assert meta.doc_type == "webpage"
        assert meta.chunk_index == 0
        assert meta.keywords == []
        assert meta.key_facts == []

    def test_unique_chunk_ids(self):
        ids = {ChunkMetadata().chunk_id for _ in range(100)}
        assert len(ids) == 100

    def test_custom_values(self):
        meta = ChunkMetadata(
            source_url="https://example.com",
            domain="car",
            section_path="Coverage > Third Party",
            page_number=5,
            language="en",
        )
        assert meta.source_url == "https://example.com"
        assert meta.domain == "car"
        assert meta.page_number == 5


class TestChunk:
    def test_content_with_context_generated(self):
        chunk = Chunk(
            content="Insurance coverage details",
            metadata=ChunkMetadata(
                source_doc_title="Car Policy",
                domain="car",
                section_path="Coverage",
            ),
        )
        assert "Document: Car Policy" in chunk.content_with_context
        assert "Domain: car" in chunk.content_with_context
        assert "Section: Coverage" in chunk.content_with_context
        assert "Insurance coverage details" in chunk.content_with_context

    def test_content_with_context_no_metadata(self):
        chunk = Chunk(content="Just text")
        assert chunk.content_with_context == "Just text"

    def test_token_count_estimated(self):
        chunk = Chunk(content="one two three four five")
        assert chunk.token_count == 10  # 5 words * 2

    def test_explicit_token_count_preserved(self):
        chunk = Chunk(content="text", token_count=42)
        assert chunk.token_count == 42

    def test_source_doc_id_generated_from_url(self):
        chunk = Chunk(
            content="text",
            metadata=ChunkMetadata(source_url="https://example.com/page"),
        )
        assert chunk.metadata.source_doc_id != ""
        assert len(chunk.metadata.source_doc_id) == 12

    def test_source_doc_id_deterministic(self):
        chunk1 = Chunk(
            content="a",
            metadata=ChunkMetadata(source_url="https://example.com/page"),
        )
        chunk2 = Chunk(
            content="b",
            metadata=ChunkMetadata(source_url="https://example.com/page"),
        )
        assert chunk1.metadata.source_doc_id == chunk2.metadata.source_doc_id

    def test_source_doc_id_not_generated_without_url(self):
        chunk = Chunk(content="text")
        assert chunk.metadata.source_doc_id == ""

    def test_explicit_content_with_context_preserved(self):
        chunk = Chunk(
            content="text",
            content_with_context="[Custom Header]\ntext",
            metadata=ChunkMetadata(source_doc_title="Doc"),
        )
        assert chunk.content_with_context == "[Custom Header]\ntext"
