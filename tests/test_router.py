"""Tests for agent.nodes.router keyword routing and LLM fallback."""

from unittest.mock import MagicMock

from agent.nodes.router import _keyword_route, router


class TestKeywordRoute:
    def test_car_english(self):
        result = _keyword_route("I need car insurance")
        assert "car" in result

    def test_car_hebrew(self):
        result = _keyword_route("ביטוח רכב")
        assert "car" in result

    def test_health_english(self):
        result = _keyword_route("I need health insurance for hospital visits")
        assert "health" in result

    def test_dental_english(self):
        result = _keyword_route("dental insurance for teeth")
        assert "dental" in result

    def test_travel_english(self):
        result = _keyword_route("travel insurance for my flight abroad")
        assert "travel" in result

    def test_travel_hebrew(self):
        result = _keyword_route("ביטוח נסיעות לחו\"ל")
        assert "travel" in result

    def test_mortgage_english(self):
        result = _keyword_route("mortgage insurance for my home loan")
        assert "mortgage" in result

    def test_business_english(self):
        result = _keyword_route("business insurance for liability")
        assert "business" in result

    def test_apartment_english(self):
        result = _keyword_route("apartment insurance for my home property contents")
        assert "apartment" in result

    def test_apartment_hebrew(self):
        result = _keyword_route("ביטוח דירה")
        assert "apartment" in result

    def test_life_english(self):
        result = _keyword_route("life insurance death benefit beneficiary")
        assert "life" in result

    def test_no_match(self):
        result = _keyword_route("what is the weather today")
        assert result == []

    def test_multiple_domains(self):
        result = _keyword_route("I need car and health insurance")
        assert "car" in result
        assert "health" in result

    def test_sorted_by_hit_count(self):
        # More keyword matches should rank higher
        result = _keyword_route("car vehicle auto insurance driving motor")
        assert result[0] == "car"

    def test_hebrew_domain_name_fallback(self):
        result = _keyword_route("מה כולל ביטוח בריאות")
        assert "health" in result

    def test_case_insensitive(self):
        result = _keyword_route("CAR INSURANCE")
        assert "car" in result


class TestRouterNode:
    def test_keyword_match_skips_llm(self):
        llm = MagicMock()
        state = {"query": "car insurance", "rewritten_query": "", "reasoning_trace": []}
        result = router(state, llm)
        assert "car" in result["detected_domains"]
        assert result["should_fallback"] is False
        llm.generate.assert_not_called()

    def test_uses_rewritten_query_if_available(self):
        llm = MagicMock()
        state = {
            "query": "tell me about it",
            "rewritten_query": "car insurance coverage",
            "reasoning_trace": [],
        }
        result = router(state, llm)
        assert "car" in result["detected_domains"]

    def test_llm_fallback_on_no_keyword_match(self):
        llm = MagicMock()
        llm.generate.return_value = "car"
        state = {"query": "what coverage do I have", "rewritten_query": "", "reasoning_trace": []}
        result = router(state, llm)
        assert "car" in result["detected_domains"]
        assert result["should_fallback"] is False

    def test_llm_off_topic(self):
        llm = MagicMock()
        llm.generate.return_value = "off_topic"
        state = {
            "query": "what is the weather today",
            "rewritten_query": "",
            "reasoning_trace": [],
        }
        result = router(state, llm)
        assert result["detected_domains"] == []
        assert result["should_fallback"] is True

    def test_llm_unrecognized_response(self):
        llm = MagicMock()
        llm.generate.return_value = "something random"
        state = {"query": "blah blah", "rewritten_query": "", "reasoning_trace": []}
        result = router(state, llm)
        assert result["detected_domains"] == []
        # "off_topic" not in response, so should_fallback is False
        assert result["should_fallback"] is False

    def test_reasoning_trace_appended(self):
        llm = MagicMock()
        state = {
            "query": "car insurance",
            "rewritten_query": "",
            "reasoning_trace": ["Previous trace"],
        }
        result = router(state, llm)
        assert len(result["reasoning_trace"]) == 2
        assert result["reasoning_trace"][0] == "Previous trace"
        assert "Router:" in result["reasoning_trace"][1]
