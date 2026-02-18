"""Score chatbot answers using existing evaluation metrics and LLM-as-judge."""

import json
import re
from dataclasses import dataclass, field

from loguru import logger

from evaluation.citation_scorer import score_citations
from llm.claude_client import ClaudeClient
from quizzer.prompts import ANSWER_SCORING_PROMPT, TYPE_SCORING_CRITERIA
from quizzer.question_types import GeneratedQuestion, QuestionType

# Per-type weight profiles for overall_score calculation.
# Keys: (correctness, completeness, relevance, citation_quality, citation_f1, tone, efficiency, domain_match, type_accuracy)
TYPE_SCORE_WEIGHTS: dict[QuestionType, dict[str, float]] = {
    QuestionType.YES_NO: {
        "correctness": 0.25, "completeness": 0.10, "relevance": 0.15,
        "citation_quality": 0.10, "citation_f1": 0.05, "tone": 0.05,
        "efficiency": 0.05, "domain_match": 0.05, "type_accuracy": 0.20,
    },
    QuestionType.NUMERICAL: {
        "correctness": 0.20, "completeness": 0.10, "relevance": 0.15,
        "citation_quality": 0.10, "citation_f1": 0.05, "tone": 0.05,
        "efficiency": 0.05, "domain_match": 0.05, "type_accuracy": 0.25,
    },
    QuestionType.CONDITIONAL: {
        "correctness": 0.20, "completeness": 0.25, "relevance": 0.10,
        "citation_quality": 0.10, "citation_f1": 0.05, "tone": 0.05,
        "efficiency": 0.05, "domain_match": 0.05, "type_accuracy": 0.15,
    },
    QuestionType.FACTUAL: {
        "correctness": 0.25, "completeness": 0.15, "relevance": 0.15,
        "citation_quality": 0.10, "citation_f1": 0.10, "tone": 0.05,
        "efficiency": 0.05, "domain_match": 0.05, "type_accuracy": 0.10,
    },
    QuestionType.COMPARISON: {
        "correctness": 0.20, "completeness": 0.15, "relevance": 0.10,
        "citation_quality": 0.10, "citation_f1": 0.05, "tone": 0.05,
        "efficiency": 0.05, "domain_match": 0.05, "type_accuracy": 0.25,
    },
    QuestionType.PROCEDURAL: {
        "correctness": 0.20, "completeness": 0.15, "relevance": 0.10,
        "citation_quality": 0.10, "citation_f1": 0.05, "tone": 0.05,
        "efficiency": 0.05, "domain_match": 0.05, "type_accuracy": 0.25,
    },
}

# Default weights (used if type is not found in the map)
DEFAULT_SCORE_WEIGHTS: dict[str, float] = {
    "correctness": 0.25, "completeness": 0.20, "relevance": 0.15,
    "citation_quality": 0.15, "citation_f1": 0.10, "tone": 0.05,
    "efficiency": 0.05, "domain_match": 0.05, "type_accuracy": 0.00,
}


@dataclass
class QuizScore:
    """Complete score for a single quiz question-answer pair."""

    # Question info
    question: str
    question_type: QuestionType
    domain: str
    language: str
    difficulty: str

    # Answer info
    answer: str
    citations: list[dict] = field(default_factory=list)
    api_domain: str | None = None
    confidence: float = 0.0
    latency_s: float = 0.0
    success: bool = True

    # LLM judge scores (0-1)
    correctness: float = 0.0
    completeness: float = 0.0
    citation_quality: float = 0.0
    relevance: float = 0.0
    tone: float = 0.0
    llm_reasoning: str = ""

    # Automated citation score
    citation_precision: float = 0.0
    citation_recall: float = 0.0
    citation_f1: float = 0.0

    # Efficiency score
    efficiency: float = 0.0

    # Domain routing accuracy
    domain_match: bool = False

    # Type-specific accuracy
    type_accuracy: float = 0.0

    @property
    def overall_score(self) -> float:
        """Weighted overall score using per-type weight profiles."""
        w = TYPE_SCORE_WEIGHTS.get(self.question_type, DEFAULT_SCORE_WEIGHTS)
        return (
            self.correctness * w["correctness"]
            + self.completeness * w["completeness"]
            + self.relevance * w["relevance"]
            + self.citation_quality * w["citation_quality"]
            + self.citation_f1 * w["citation_f1"]
            + self.tone * w["tone"]
            + self.efficiency * w["efficiency"]
            + (1.0 if self.domain_match else 0.0) * w["domain_match"]
            + self.type_accuracy * w["type_accuracy"]
        )

    def to_dict(self) -> dict:
        """Serialize to dict for JSON export."""
        return {
            "question": self.question,
            "question_type": self.question_type.value,
            "domain": self.domain,
            "language": self.language,
            "difficulty": self.difficulty,
            "answer": self.answer[:500],
            "num_citations": len(self.citations),
            "api_domain": self.api_domain,
            "confidence": self.confidence,
            "latency_s": self.latency_s,
            "success": self.success,
            "correctness": round(self.correctness, 3),
            "completeness": round(self.completeness, 3),
            "citation_quality": round(self.citation_quality, 3),
            "relevance": round(self.relevance, 3),
            "tone": round(self.tone, 3),
            "llm_reasoning": self.llm_reasoning,
            "citation_precision": round(self.citation_precision, 3),
            "citation_recall": round(self.citation_recall, 3),
            "citation_f1": round(self.citation_f1, 3),
            "efficiency": round(self.efficiency, 3),
            "domain_match": self.domain_match,
            "type_accuracy": round(self.type_accuracy, 3),
            "overall_score": round(self.overall_score, 3),
        }


def _parse_llm_scores(response: str) -> dict:
    """Parse JSON scores from LLM response."""
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        match = re.search(r"\{[^{}]+\}", response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

    # Last resort: regex extraction
    scores = {}
    for key in ("correctness", "completeness", "citation_quality", "relevance", "tone", "type_accuracy"):
        m = re.search(rf'"{key}"\s*:\s*([\d.]+)', response)
        if m:
            scores[key] = max(0.0, min(1.0, float(m.group(1))))
    return scores


def score_answer(
    question: GeneratedQuestion,
    api_response: dict,
    client: ClaudeClient,
) -> QuizScore:
    """Score a chatbot answer against the generated question.

    Uses both automated metrics (citation scoring, efficiency) and
    LLM-as-judge scoring (correctness, completeness, relevance, tone).

    Args:
        question: The generated question with source documents.
        api_response: The chatbot API response dict.
        client: ClaudeClient for LLM judge calls.

    Returns:
        QuizScore with all dimensions scored.
    """
    answer = api_response.get("answer", "")
    citations = api_response.get("citations", [])
    latency = api_response.get("latency_s", 0.0)
    api_domain = api_response.get("domain")

    score = QuizScore(
        question=question.question,
        question_type=question.question_type,
        domain=question.domain,
        language=question.language,
        difficulty=question.difficulty,
        answer=answer,
        citations=citations,
        api_domain=api_domain,
        confidence=api_response.get("confidence", 0.0),
        latency_s=latency,
        success=api_response.get("success", False),
    )

    if not api_response.get("success") or not answer:
        return score

    # 1. Automated citation scoring
    source_docs = [
        {
            "source_url": d.get("source_url", ""),
            "source_doc_title": d.get("source_doc_title", ""),
        }
        for d in question.source_documents
    ]
    cit_scores = score_citations(answer, citations, source_docs)
    score.citation_precision = cit_scores.get("precision", 0.0)
    score.citation_recall = cit_scores.get("recall", 0.0)
    score.citation_f1 = cit_scores.get("score", 0.0)

    # 2. Efficiency score (target: <5s=1.0, >30s=0.0)
    score.efficiency = max(0.0, min(1.0, 1.0 - (latency - 5.0) / 25.0))

    # 3. Domain routing accuracy
    score.domain_match = (api_domain == question.domain) if api_domain else False

    # 4. LLM-as-judge scoring
    citations_text = "\n".join(
        f"- {c.get('document_title', 'N/A')}: {c.get('relevant_text', 'N/A')[:200]}"
        for c in citations
    ) or "(no citations provided)"

    source_docs_text = "\n".join(
        f"- [{d.get('domain', '')}] {d.get('source_doc_title', 'N/A')}: "
        f"{d.get('content', '')[:300]}"
        for d in question.source_documents
    )

    type_criteria = TYPE_SCORING_CRITERIA.get(
        question.question_type.value,
        "   Score based on whether the answer format matches what the question expects.\n"
        "   - 1.0: Perfect format match\n"
        "   - 0.5: Partially matches expected format\n"
        "   - 0.0: Does not match expected format at all",
    )

    prompt = ANSWER_SCORING_PROMPT.format(
        question=question.question,
        question_type=question.question_type.value,
        expected_hints=question.expected_answer_hints,
        answer=answer[:2000],
        citations=citations_text,
        source_docs=source_docs_text[:3000],
        type_specific_criteria=type_criteria,
    )

    try:
        response = client.generate(prompt, temperature=0.0, max_tokens=512)
        scores = _parse_llm_scores(response)

        score.correctness = scores.get("correctness", 0.0)
        score.completeness = scores.get("completeness", 0.0)
        score.citation_quality = scores.get("citation_quality", 0.0)
        score.relevance = scores.get("relevance", 0.0)
        score.tone = scores.get("tone", 0.0)
        score.type_accuracy = scores.get("type_accuracy", 0.0)
        score.llm_reasoning = scores.get("reasoning", "")

    except Exception as e:
        logger.error(f"LLM scoring failed: {e}")

    return score
