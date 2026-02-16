"""Custom citation accuracy scorer."""

import re


def score_citations(
    answer: str,
    citations: list[dict],
    source_documents: list[dict],
) -> dict:
    """Score citation quality for a generated answer.

    Returns:
        Dict with precision, recall, and overall citation score.
    """
    if not citations:
        # If the answer makes factual claims but has no citations, that's bad
        has_claims = len(answer.split()) > 20  # rough heuristic
        return {
            "precision": 0.0,
            "recall": 0.0,
            "score": 0.0 if has_claims else 1.0,  # No claims = no citations needed
        }

    # Check citation precision: are cited sources real and relevant?
    valid_citations = 0
    for cit in citations:
        url = cit.get("source_url", "")
        title = cit.get("document_title", "")

        # Check if this citation matches a source document
        for doc in source_documents:
            if (
                (url and url == doc.get("source_url", ""))
                or (title and title == doc.get("source_doc_title", ""))
            ):
                valid_citations += 1
                break

    precision = valid_citations / len(citations) if citations else 0.0

    # Check citation recall: do factual claims in the answer have citations?
    # Count sentences that look like factual claims
    sentences = re.split(r"[.!?。]\s+", answer)
    factual_sentences = [
        s for s in sentences
        if len(s.split()) > 5 and not s.strip().startswith(("אני", "I ", "Please", "בבקשה"))
    ]

    # Check how many have citation references nearby
    cited_sentences = 0
    for s in factual_sentences:
        # Look for citation markers in or near the sentence
        if re.search(r"\[(?:Source|מקור|[0-9]+)\]", s) or re.search(r"\[\d+\]", s):
            cited_sentences += 1

    recall = cited_sentences / len(factual_sentences) if factual_sentences else 1.0

    # F1-like combined score
    if precision + recall > 0:
        score = 2 * precision * recall / (precision + recall)
    else:
        score = 0.0

    return {
        "precision": precision,
        "recall": recall,
        "score": score,
        "total_citations": len(citations),
        "valid_citations": valid_citations,
        "factual_sentences": len(factual_sentences),
        "cited_sentences": cited_sentences,
    }
