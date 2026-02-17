"""Keyword-based evaluation: checks if generated answers contain required/forbidden terms."""

import re
from dataclasses import dataclass


@dataclass
class KeywordScore:
    """Result of keyword evaluation for a single question."""
    precision: float  # What fraction of mentioned keywords are actually required
    recall: float  # What fraction of required keywords are mentioned
    forbidden_count: int  # Number of forbidden keywords found in answer
    matched_keywords: list[str]
    missing_keywords: list[str]
    found_forbidden: list[str]
    score: float  # Combined score (0-1)


def _normalize(text: str) -> str:
    """Normalize text for keyword matching (lowercase, strip punctuation)."""
    return re.sub(r"[^\w\s]", "", text.lower().strip())


def score_keywords(
    answer: str,
    required_keywords: list[str],
    forbidden_keywords: list[str] | None = None,
) -> KeywordScore:
    """Score an answer based on required and forbidden keyword presence.

    Args:
        answer: The generated answer text.
        required_keywords: Keywords that SHOULD appear in a correct answer.
        forbidden_keywords: Keywords that should NOT appear (e.g., competitor names).

    Returns:
        KeywordScore with precision, recall, forbidden count, and combined score.
    """
    forbidden_keywords = forbidden_keywords or []
    answer_norm = _normalize(answer)

    # Check required keywords
    matched = []
    missing = []
    for kw in required_keywords:
        kw_norm = _normalize(kw)
        if kw_norm and kw_norm in answer_norm:
            matched.append(kw)
        else:
            missing.append(kw)

    # Check forbidden keywords
    found_forbidden = []
    for kw in forbidden_keywords:
        kw_norm = _normalize(kw)
        if kw_norm and kw_norm in answer_norm:
            found_forbidden.append(kw)

    # Calculate scores
    recall = len(matched) / len(required_keywords) if required_keywords else 1.0
    precision = len(matched) / (len(matched) + len(found_forbidden)) if (matched or found_forbidden) else 1.0

    # Combined score: recall weighted heavily, with penalty for forbidden terms
    forbidden_penalty = 0.1 * len(found_forbidden)
    score = max(0.0, recall * 0.9 + precision * 0.1 - forbidden_penalty)

    return KeywordScore(
        precision=round(precision, 3),
        recall=round(recall, 3),
        forbidden_count=len(found_forbidden),
        matched_keywords=matched,
        missing_keywords=missing,
        found_forbidden=found_forbidden,
        score=round(score, 3),
    )


def score_keywords_batch(
    results: list[dict],
) -> dict:
    """Score a batch of evaluation results that include keyword annotations.

    Args:
        results: List of dicts with keys: "answer", "required_keywords", "forbidden_keywords" (optional).

    Returns:
        Aggregate keyword scores.
    """
    scores = []
    for r in results:
        s = score_keywords(
            answer=r["answer"],
            required_keywords=r.get("required_keywords", []),
            forbidden_keywords=r.get("forbidden_keywords", []),
        )
        scores.append(s)

    if not scores:
        return {"avg_recall": 0.0, "avg_precision": 0.0, "avg_score": 0.0, "total_forbidden": 0}

    return {
        "avg_recall": round(sum(s.recall for s in scores) / len(scores), 3),
        "avg_precision": round(sum(s.precision for s in scores) / len(scores), 3),
        "avg_score": round(sum(s.score for s in scores) / len(scores), 3),
        "total_forbidden": sum(s.forbidden_count for s in scores),
    }
