"""Tests for agent.nodes.fallback."""

from agent.nodes.fallback import FALLBACK_EN, FALLBACK_HE, fallback


class TestFallback:
    def test_hebrew_default(self):
        state = {"detected_language": "he", "graded_documents": []}
        result = fallback(state)
        assert result["generation"] == FALLBACK_HE
        assert result["citations"] == []
        assert result["should_fallback"] is True

    def test_english(self):
        state = {"detected_language": "en", "graded_documents": []}
        result = fallback(state)
        assert result["generation"] == FALLBACK_EN

    def test_defaults_to_hebrew(self):
        state = {}
        result = fallback(state)
        assert FALLBACK_HE in result["generation"]

    def test_includes_partial_docs_when_available(self):
        docs = [
            {
                "source_doc_title": "Car Insurance",
                "section_path": "Coverage",
                "content": "Your car insurance covers...",
            }
        ]
        state = {"detected_language": "en", "graded_documents": docs}
        result = fallback(state)
        assert "Car Insurance" in result["generation"]
        assert "Coverage" in result["generation"]
        assert "might be relevant" in result["generation"]

    def test_limits_partial_docs_to_two(self):
        docs = [
            {"source_doc_title": f"Doc {i}", "section_path": "S", "content": f"Content {i}"}
            for i in range(5)
        ]
        state = {"detected_language": "en", "graded_documents": docs}
        result = fallback(state)
        assert "Doc 0" in result["generation"]
        assert "Doc 1" in result["generation"]
        assert "Doc 2" not in result["generation"]

    def test_hebrew_partial_info_label(self):
        docs = [{"source_doc_title": "ביטוח", "section_path": "", "content": "תוכן"}]
        state = {"detected_language": "he", "graded_documents": docs}
        result = fallback(state)
        assert "עם זאת" in result["generation"]

    def test_contact_info_present(self):
        state = {"detected_language": "en"}
        result = fallback(state)
        assert "*6060" in result["generation"]
        assert "harel-group.co.il" in result["generation"]
