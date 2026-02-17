"""Tests for evaluation.keyword_scorer."""

from evaluation.keyword_scorer import KeywordScore, score_keywords, score_keywords_batch


class TestScoreKeywords:
    def test_all_keywords_found(self):
        answer = "The car insurance covers third party liability."
        result = score_keywords(answer, ["car", "insurance", "liability"])
        assert result.recall == 1.0
        assert len(result.matched_keywords) == 3
        assert len(result.missing_keywords) == 0

    def test_no_keywords_found(self):
        answer = "Hello world."
        result = score_keywords(answer, ["insurance", "coverage", "premium"])
        assert result.recall == 0.0
        assert len(result.missing_keywords) == 3

    def test_partial_match(self):
        answer = "The insurance is comprehensive."
        result = score_keywords(answer, ["insurance", "premium", "deductible"])
        assert result.recall == pytest.approx(1 / 3, abs=0.01)
        assert "insurance" in result.matched_keywords
        assert "premium" in result.missing_keywords

    def test_forbidden_keyword_detected(self):
        answer = "Our competitor AIG offers cheaper rates."
        result = score_keywords(answer, ["rates"], forbidden_keywords=["AIG", "Migdal"])
        assert result.forbidden_count == 1
        assert "AIG" in result.found_forbidden

    def test_no_forbidden_keywords(self):
        answer = "Harel insurance is great."
        result = score_keywords(answer, ["insurance"], forbidden_keywords=["AIG"])
        assert result.forbidden_count == 0
        assert result.found_forbidden == []

    def test_case_insensitive(self):
        answer = "The INSURANCE covers LIABILITY."
        result = score_keywords(answer, ["insurance", "liability"])
        assert result.recall == 1.0

    def test_empty_required_keywords(self):
        result = score_keywords("Any answer", [])
        assert result.recall == 1.0

    def test_empty_answer(self):
        result = score_keywords("", ["insurance"])
        assert result.recall == 0.0

    def test_score_combines_precision_and_recall(self):
        # Full match, no forbidden -> score should be high
        result = score_keywords("insurance policy", ["insurance", "policy"])
        assert result.score > 0.8

    def test_forbidden_penalty(self):
        result = score_keywords("AIG insurance", ["insurance"], forbidden_keywords=["AIG"])
        # Has forbidden keyword, should reduce score
        assert result.score < score_keywords("Harel insurance", ["insurance"]).score


class TestScoreKeywordsBatch:
    def test_empty_batch(self):
        result = score_keywords_batch([])
        assert result["avg_recall"] == 0.0
        assert result["avg_score"] == 0.0

    def test_single_result(self):
        results = [
            {
                "answer": "The car insurance covers damages.",
                "required_keywords": ["car", "insurance", "damages"],
            }
        ]
        result = score_keywords_batch(results)
        assert result["avg_recall"] == 1.0

    def test_multiple_results(self):
        results = [
            {"answer": "car insurance", "required_keywords": ["car", "insurance"]},
            {"answer": "health plan", "required_keywords": ["health", "plan"]},
        ]
        result = score_keywords_batch(results)
        assert result["avg_recall"] == 1.0

    def test_with_forbidden(self):
        results = [
            {
                "answer": "AIG competitor is bad",
                "required_keywords": ["competitor"],
                "forbidden_keywords": ["AIG"],
            }
        ]
        result = score_keywords_batch(results)
        assert result["total_forbidden"] == 1


import pytest  # noqa: E402
