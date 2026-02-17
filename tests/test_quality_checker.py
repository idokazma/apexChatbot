"""Tests for agent.nodes.quality_checker."""

from unittest.mock import MagicMock

from agent.nodes.quality_checker import quality_checker


class TestQualityChecker:
    def _make_state(self, **overrides):
        base = {
            "generation": "The insurance covers damages up to 1M NIS.",
            "query": "What does car insurance cover?",
            "graded_documents": [
                {"source_doc_title": "Car Policy", "content": "Coverage includes..."}
            ],
            "detected_domains": ["car"],
        }
        base.update(overrides)
        return base

    def test_pass(self):
        llm = MagicMock()
        llm.generate.return_value = "PASS\nReasoning: Answer is well-grounded."
        result = quality_checker(self._make_state(), llm)
        assert result["is_grounded"] is True
        assert result["quality_action"] == "pass"
        assert result["quality_reasoning"] != ""

    def test_reroute(self):
        llm = MagicMock()
        llm.generate.return_value = "REROUTE: health\nReasoning: Wrong domain."
        result = quality_checker(self._make_state(), llm)
        assert result["is_grounded"] is False
        assert result["quality_action"] == "reroute"
        assert result["detected_domains"] == ["health"]

    def test_rephrase(self):
        llm = MagicMock()
        llm.generate.return_value = "REPHRASE: What specific coverage does car insurance provide?\nReasoning: Too vague."
        result = quality_checker(self._make_state(), llm)
        assert result["is_grounded"] is False
        assert result["quality_action"] == "rephrase"
        assert "coverage" in result["rewritten_query"].lower()

    def test_rephrase_query_on_second_line(self):
        llm = MagicMock()
        llm.generate.return_value = "REPHRASE:\nWhat are the deductibles for car insurance?\nReasoning: Missing info."
        result = quality_checker(self._make_state(), llm)
        assert result["quality_action"] == "rephrase"
        assert "deductibles" in result["rewritten_query"].lower()

    def test_fail_on_unparseable(self):
        llm = MagicMock()
        llm.generate.return_value = "Some random garbage output"
        result = quality_checker(self._make_state(), llm)
        assert result["is_grounded"] is False
        assert result["quality_action"] == "fail"

    def test_no_answer(self):
        llm = MagicMock()
        result = quality_checker(self._make_state(generation=""), llm)
        assert result["quality_action"] == "fail"
        llm.generate.assert_not_called()

    def test_no_documents(self):
        llm = MagicMock()
        result = quality_checker(self._make_state(graded_documents=[]), llm)
        assert result["quality_action"] == "fail"
        llm.generate.assert_not_called()

    def test_empty_state(self):
        llm = MagicMock()
        result = quality_checker({}, llm)
        assert result["quality_action"] == "fail"
        assert result["is_grounded"] is False
