"""Tests for agent.graph conditional edge functions and increment_retry."""

from agent.graph import (
    _grade_decision,
    _increment_retry,
    _quality_decision,
    _should_fallback_after_route,
)


class TestShouldFallbackAfterRoute:
    def test_fallback_when_should_fallback_true(self):
        state = {"should_fallback": True}
        assert _should_fallback_after_route(state) == "fallback"

    def test_retrieve_when_should_fallback_false(self):
        state = {"should_fallback": False}
        assert _should_fallback_after_route(state) == "retrieve"

    def test_retrieve_when_should_fallback_missing(self):
        state = {}
        assert _should_fallback_after_route(state) == "retrieve"


class TestGradeDecision:
    def test_generate_when_graded_docs_exist(self):
        state = {"graded_documents": [{"content": "doc1"}], "retry_count": 0}
        assert _grade_decision(state) == "generate"

    def test_retry_when_no_docs_and_under_limit(self):
        state = {"graded_documents": [], "retry_count": 1}
        assert _grade_decision(state) == "retry"

    def test_fallback_when_no_docs_and_at_limit(self):
        state = {"graded_documents": [], "retry_count": 3}
        assert _grade_decision(state) == "fallback"

    def test_fallback_when_no_docs_and_over_limit(self):
        state = {"graded_documents": [], "retry_count": 5}
        assert _grade_decision(state) == "fallback"

    def test_generate_with_single_doc(self):
        state = {"graded_documents": [{"content": "one"}], "retry_count": 0}
        assert _grade_decision(state) == "generate"

    def test_retry_at_count_2(self):
        state = {"graded_documents": [], "retry_count": 2}
        assert _grade_decision(state) == "retry"

    def test_missing_fields_default(self):
        state = {}
        # graded_documents defaults to [], retry_count defaults to 0
        assert _grade_decision(state) == "retry"


class TestQualityDecision:
    def test_pass_ends(self):
        state = {"quality_action": "pass", "retry_count": 0}
        assert _quality_decision(state) == "end"

    def test_reroute_retries(self):
        state = {"quality_action": "reroute", "retry_count": 1}
        assert _quality_decision(state) == "retry"

    def test_rephrase_retries(self):
        state = {"quality_action": "rephrase", "retry_count": 0}
        assert _quality_decision(state) == "retry"

    def test_reroute_at_limit_falls_back(self):
        state = {"quality_action": "reroute", "retry_count": 3}
        assert _quality_decision(state) == "fallback"

    def test_rephrase_at_limit_falls_back(self):
        state = {"quality_action": "rephrase", "retry_count": 3}
        assert _quality_decision(state) == "fallback"

    def test_fail_falls_back(self):
        state = {"quality_action": "fail", "retry_count": 0}
        assert _quality_decision(state) == "fallback"

    def test_unknown_action_falls_back(self):
        state = {"quality_action": "unknown", "retry_count": 0}
        assert _quality_decision(state) == "fallback"

    def test_missing_action_falls_back(self):
        state = {}
        assert _quality_decision(state) == "fallback"


class TestIncrementRetry:
    def test_increments_count(self):
        state = {
            "quality_action": "reroute",
            "retry_count": 0,
            "query": "original query",
            "quality_reasoning": "wrong domain",
            "reasoning_trace": [],
        }
        result = _increment_retry(state)
        assert result["retry_count"] == 1

    def test_preserves_query_on_non_rephrase(self):
        state = {
            "quality_action": "reroute",
            "retry_count": 0,
            "query": "original query",
            "quality_reasoning": "",
            "reasoning_trace": [],
        }
        result = _increment_retry(state)
        assert result["rewritten_query"] == "original query"

    def test_does_not_overwrite_on_rephrase(self):
        state = {
            "quality_action": "rephrase",
            "retry_count": 1,
            "query": "original query",
            "quality_reasoning": "needs rephrasing",
            "reasoning_trace": [],
        }
        result = _increment_retry(state)
        assert "rewritten_query" not in result
        assert result["retry_count"] == 2

    def test_appends_reasoning_trace(self):
        state = {
            "quality_action": "reroute",
            "retry_count": 0,
            "query": "q",
            "quality_reasoning": "test reason",
            "reasoning_trace": ["existing"],
        }
        result = _increment_retry(state)
        assert len(result["reasoning_trace"]) == 2
        assert "existing" in result["reasoning_trace"]
        assert "Retry #1 (reroute)" in result["reasoning_trace"][1]
