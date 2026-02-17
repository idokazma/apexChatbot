"""Tests for agent.nodes.grader."""

from unittest.mock import MagicMock

from agent.nodes.grader import grader


class TestGrader:
    def test_no_documents(self):
        llm = MagicMock()
        state = {"query": "test", "retrieved_documents": [], "retry_count": 0, "reasoning_trace": []}
        result = grader(state, llm)
        assert result["graded_documents"] == []
        llm.generate.assert_not_called()

    def test_relevant_document_kept(self):
        llm = MagicMock()
        llm.generate.return_value = "yes"
        state = {
            "query": "car insurance",
            "retrieved_documents": [
                {"content": "Car insurance covers...", "source_doc_title": "Car Doc"}
            ],
            "retry_count": 0,
            "reasoning_trace": [],
        }
        result = grader(state, llm)
        assert len(result["graded_documents"]) == 1
        assert result["graded_documents"][0]["is_relevant"] is True

    def test_irrelevant_document_rejected(self):
        llm = MagicMock()
        llm.generate.return_value = "no"
        state = {
            "query": "car insurance",
            "retrieved_documents": [
                {"content": "Unrelated content", "source_doc_title": "Bad Doc"}
            ],
            "retry_count": 0,
            "reasoning_trace": [],
        }
        result = grader(state, llm)
        assert len(result["graded_documents"]) == 0

    def test_mixed_documents(self):
        llm = MagicMock()
        llm.generate.side_effect = ["yes", "no", "yes"]
        state = {
            "query": "car insurance",
            "retrieved_documents": [
                {"content": "Relevant 1", "source_doc_title": "Doc 1"},
                {"content": "Irrelevant", "source_doc_title": "Doc 2"},
                {"content": "Relevant 2", "source_doc_title": "Doc 3"},
            ],
            "retry_count": 0,
            "reasoning_trace": [],
        }
        result = grader(state, llm)
        assert len(result["graded_documents"]) == 2

    def test_reasoning_trace_appended(self):
        llm = MagicMock()
        llm.generate.return_value = "yes"
        state = {
            "query": "q",
            "retrieved_documents": [{"content": "c", "source_doc_title": "d"}],
            "retry_count": 0,
            "reasoning_trace": ["prev"],
        }
        result = grader(state, llm)
        assert len(result["reasoning_trace"]) == 2
        assert "Grader: 1/1 relevant" in result["reasoning_trace"][1]

    def test_content_truncated_in_prompt(self):
        llm = MagicMock()
        llm.generate.return_value = "yes"
        long_content = "x" * 5000
        state = {
            "query": "q",
            "retrieved_documents": [{"content": long_content, "source_doc_title": "d"}],
            "retry_count": 0,
            "reasoning_trace": [],
        }
        grader(state, llm)
        # Verify the prompt was called with truncated content
        call_args = llm.generate.call_args
        prompt = call_args[0][0]
        assert len(prompt) < len(long_content)
