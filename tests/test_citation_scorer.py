"""Tests for evaluation.citation_scorer."""

from evaluation.citation_scorer import score_citations


class TestScoreCitations:
    def test_no_citations_with_claims(self):
        # Answer must exceed 20 words to be considered as having claims
        answer = (
            "The insurance policy covers all types of property damage including fire and flood. "
            "It also provides coverage for theft, vandalism, and natural disasters in most cases."
        )
        result = score_citations(answer, [], [])
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["score"] == 0.0

    def test_no_citations_short_answer(self):
        answer = "Yes, it does."
        result = score_citations(answer, [], [])
        assert result["score"] == 1.0  # No claims = no citations needed

    def test_valid_citation_by_url(self):
        answer = "Coverage includes [1] fire damage."
        citations = [{"source_url": "https://a.com", "document_title": "Doc A"}]
        source_docs = [{"source_url": "https://a.com", "source_doc_title": "Doc A"}]
        result = score_citations(answer, citations, source_docs)
        assert result["precision"] == 1.0
        assert result["valid_citations"] == 1

    def test_valid_citation_by_title(self):
        answer = "Coverage includes [1] fire damage."
        citations = [{"source_url": "", "document_title": "Doc A"}]
        source_docs = [{"source_url": "", "source_doc_title": "Doc A"}]
        result = score_citations(answer, citations, source_docs)
        assert result["precision"] == 1.0

    def test_invalid_citation(self):
        answer = "Coverage includes [1] fire damage."
        citations = [{"source_url": "https://fake.com", "document_title": "Fake Doc"}]
        source_docs = [{"source_url": "https://real.com", "source_doc_title": "Real Doc"}]
        result = score_citations(answer, citations, source_docs)
        assert result["precision"] == 0.0
        assert result["valid_citations"] == 0

    def test_mixed_valid_invalid(self):
        answer = "According to [1] and [2], the coverage is good."
        citations = [
            {"source_url": "https://a.com", "document_title": "Doc A"},
            {"source_url": "https://fake.com", "document_title": "Fake"},
        ]
        source_docs = [{"source_url": "https://a.com", "source_doc_title": "Doc A"}]
        result = score_citations(answer, citations, source_docs)
        assert result["precision"] == 0.5
        assert result["valid_citations"] == 1
        assert result["total_citations"] == 2

    def test_recall_with_cited_sentences(self):
        # Sentences must have >5 words to be considered factual claims
        answer = (
            "The comprehensive car insurance policy covers fire damage to your vehicle [1]. "
            "It also provides full coverage for flood damage and water incidents [2]. "
            "You should call for details."
        )
        citations = [
            {"source_url": "https://a.com", "document_title": "Doc A"},
            {"source_url": "https://b.com", "document_title": "Doc B"},
        ]
        source_docs = [
            {"source_url": "https://a.com", "source_doc_title": "Doc A"},
            {"source_url": "https://b.com", "source_doc_title": "Doc B"},
        ]
        result = score_citations(answer, citations, source_docs)
        assert result["cited_sentences"] >= 1

    def test_empty_answer(self):
        result = score_citations("", [], [])
        assert result["score"] == 1.0  # No claims
