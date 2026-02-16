"""Aggregate scoring matching the competition criteria."""

from dataclasses import dataclass


@dataclass
class EvalResult:
    """Result of evaluating a single question."""

    question: str
    expected_answer: str
    generated_answer: str
    citations: list[dict]
    domain: str

    # Scores (0-1)
    relevance: float = 0.0
    citation_accuracy: float = 0.0
    efficiency: float = 0.0
    conversational_quality: float = 0.0

    @property
    def weighted_score(self) -> float:
        """Calculate weighted score matching competition criteria."""
        return (
            self.relevance * 0.65
            + self.citation_accuracy * 0.15
            + self.efficiency * 0.10
            + self.conversational_quality * 0.10
        )


def aggregate_scores(results: list[EvalResult]) -> dict:
    """Aggregate evaluation results into summary statistics."""
    if not results:
        return {}

    n = len(results)
    return {
        "total_questions": n,
        "avg_relevance": sum(r.relevance for r in results) / n,
        "avg_citation_accuracy": sum(r.citation_accuracy for r in results) / n,
        "avg_efficiency": sum(r.efficiency for r in results) / n,
        "avg_conversational_quality": sum(r.conversational_quality for r in results) / n,
        "avg_weighted_score": sum(r.weighted_score for r in results) / n,
        "per_domain": _per_domain_scores(results),
    }


def _per_domain_scores(results: list[EvalResult]) -> dict:
    """Break down scores by domain."""
    domains: dict[str, list[EvalResult]] = {}
    for r in results:
        domains.setdefault(r.domain, []).append(r)

    return {
        domain: {
            "count": len(items),
            "avg_relevance": sum(r.relevance for r in items) / len(items),
            "avg_weighted_score": sum(r.weighted_score for r in items) / len(items),
        }
        for domain, items in domains.items()
    }
