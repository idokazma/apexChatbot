"""Tests for evaluation.metrics."""

import pytest

from evaluation.metrics import EvalResult, aggregate_scores


class TestEvalResult:
    def test_weighted_score_all_perfect(self):
        r = EvalResult(
            question="q",
            expected_answer="a",
            generated_answer="a",
            citations=[],
            domain="car",
            relevance=1.0,
            citation_accuracy=1.0,
            efficiency=1.0,
            conversational_quality=1.0,
        )
        assert r.weighted_score == pytest.approx(1.0)

    def test_weighted_score_all_zero(self):
        r = EvalResult(
            question="q",
            expected_answer="a",
            generated_answer="a",
            citations=[],
            domain="car",
        )
        assert r.weighted_score == 0.0

    def test_weighted_score_relevance_dominant(self):
        r = EvalResult(
            question="q",
            expected_answer="a",
            generated_answer="a",
            citations=[],
            domain="car",
            relevance=1.0,
            citation_accuracy=0.0,
            efficiency=0.0,
            conversational_quality=0.0,
        )
        assert r.weighted_score == pytest.approx(0.65)

    def test_weighted_score_formula(self):
        r = EvalResult(
            question="q",
            expected_answer="a",
            generated_answer="a",
            citations=[],
            domain="car",
            relevance=0.8,
            citation_accuracy=0.6,
            efficiency=0.5,
            conversational_quality=0.7,
        )
        expected = 0.8 * 0.65 + 0.6 * 0.15 + 0.5 * 0.10 + 0.7 * 0.10
        assert r.weighted_score == pytest.approx(expected)


class TestAggregateScores:
    def test_empty_results(self):
        assert aggregate_scores([]) == {}

    def test_single_result(self):
        r = EvalResult(
            question="q",
            expected_answer="a",
            generated_answer="a",
            citations=[],
            domain="car",
            relevance=0.8,
            citation_accuracy=0.6,
            efficiency=0.5,
            conversational_quality=0.7,
        )
        agg = aggregate_scores([r])
        assert agg["total_questions"] == 1
        assert agg["avg_relevance"] == pytest.approx(0.8)
        assert agg["avg_citation_accuracy"] == pytest.approx(0.6)

    def test_multiple_results(self):
        results = [
            EvalResult(
                question="q1", expected_answer="a", generated_answer="a",
                citations=[], domain="car", relevance=0.8,
            ),
            EvalResult(
                question="q2", expected_answer="a", generated_answer="a",
                citations=[], domain="car", relevance=0.6,
            ),
        ]
        agg = aggregate_scores(results)
        assert agg["total_questions"] == 2
        assert agg["avg_relevance"] == pytest.approx(0.7)

    def test_per_domain_breakdown(self):
        results = [
            EvalResult(
                question="q1", expected_answer="a", generated_answer="a",
                citations=[], domain="car", relevance=0.9,
            ),
            EvalResult(
                question="q2", expected_answer="a", generated_answer="a",
                citations=[], domain="health", relevance=0.7,
            ),
        ]
        agg = aggregate_scores(results)
        per_domain = agg["per_domain"]
        assert "car" in per_domain
        assert "health" in per_domain
        assert per_domain["car"]["count"] == 1
        assert per_domain["car"]["avg_relevance"] == pytest.approx(0.9)
        assert per_domain["health"]["avg_relevance"] == pytest.approx(0.7)
