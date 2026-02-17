"""Tests for agent.nodes.generator helper functions."""

from agent.nodes.generator import _extract_citations, _format_context


class TestFormatContext:
    def test_single_document(self):
        docs = [{"source_doc_title": "My Doc", "section_path": "Section 1", "content": "Hello"}]
        result = _format_context(docs)
        assert "[1]" in result
        assert "Document: My Doc" in result
        assert "Section: Section 1" in result

    def test_multiple_documents(self):
        docs = [
            {"source_doc_title": "Doc A", "content": "Content A"},
            {"source_doc_title": "Doc B", "content": "Content B"},
        ]
        result = _format_context(docs)
        assert "[1]" in result
        assert "[2]" in result
        assert "Doc A" in result
        assert "Doc B" in result

    def test_includes_page_number(self):
        docs = [{"page_number": 5, "content": "text"}]
        result = _format_context(docs)
        assert "Page: 5" in result

    def test_includes_source_url(self):
        docs = [{"source_url": "https://example.com/doc", "content": "text"}]
        result = _format_context(docs)
        assert "URL: https://example.com/doc" in result

    def test_uses_content_expanded_when_available(self):
        docs = [{"content": "short", "content_expanded": "expanded content with neighbors"}]
        result = _format_context(docs)
        assert "expanded content with neighbors" in result
        assert "short" not in result

    def test_falls_back_to_content(self):
        docs = [{"content": "regular content"}]
        result = _format_context(docs)
        assert "regular content" in result

    def test_no_metadata_uses_document_number(self):
        docs = [{"content": "text"}]
        result = _format_context(docs)
        assert "Document 1" in result

    def test_empty_list(self):
        result = _format_context([])
        assert result == ""


class TestExtractCitations:
    def test_numbered_references(self):
        text = "According to [1], the coverage includes... Also see [2]."
        docs = [
            {"source_url": "https://a.com", "source_doc_title": "Doc A", "content": "c1"},
            {"source_url": "https://b.com", "source_doc_title": "Doc B", "content": "c2"},
        ]
        citations = _extract_citations(text, docs)
        assert len(citations) == 2
        urls = {c["source_url"] for c in citations}
        assert "https://a.com" in urls
        assert "https://b.com" in urls

    def test_out_of_range_reference_ignored(self):
        text = "According to [1] and [5]."
        docs = [{"source_url": "https://a.com", "content": "c1"}]
        citations = _extract_citations(text, docs)
        assert len(citations) == 1

    def test_no_references(self):
        text = "This answer has no citations."
        docs = [{"source_url": "https://a.com", "content": "c1"}]
        citations = _extract_citations(text, docs)
        assert len(citations) == 0

    def test_duplicate_references_deduped(self):
        text = "See [1] for details. As stated in [1]."
        docs = [{"source_url": "https://a.com", "content": "c1"}]
        citations = _extract_citations(text, docs)
        assert len(citations) == 1

    def test_source_pattern_hebrew(self):
        text = "According to [מקור: Doc A, Section 1]."
        docs = [
            {"source_url": "https://a.com", "source_doc_title": "Doc A", "content": "c1"},
        ]
        citations = _extract_citations(text, docs)
        assert len(citations) == 1
        assert citations[0]["document_title"] == "Doc A"

    def test_source_pattern_english(self):
        text = "According to [Source: Car Insurance, Coverage]."
        docs = [
            {
                "source_url": "https://a.com",
                "source_doc_title": "Car Insurance",
                "content": "c1",
            },
        ]
        citations = _extract_citations(text, docs)
        assert len(citations) == 1

    def test_citation_includes_relevant_text(self):
        text = "See [1]."
        docs = [
            {
                "source_url": "https://a.com",
                "source_doc_title": "Doc",
                "section_path": "Coverage",
                "content": "Long content text here",
                "page_number": 3,
                "source_file_path": "car/files/doc.pdf",
            }
        ]
        citations = _extract_citations(text, docs)
        assert citations[0]["relevant_text"] == "Long content text here"
        assert citations[0]["page_number"] == 3
        assert citations[0]["section"] == "Coverage"
        assert citations[0]["source_file_path"] == "car/files/doc.pdf"

    def test_truncates_relevant_text(self):
        text = "See [1]."
        long_content = "a" * 500
        docs = [{"source_url": "https://a.com", "content": long_content}]
        citations = _extract_citations(text, docs)
        assert len(citations[0]["relevant_text"]) == 200

    def test_doc_without_url_uses_index_key(self):
        text = "See [1]."
        docs = [{"content": "content"}]
        citations = _extract_citations(text, docs)
        assert len(citations) == 1
