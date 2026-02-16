"""LLM-as-judge scorers for relevance and conversational quality."""

import re

from loguru import logger

from llm.claude_client import ClaudeClient

_RELEVANCE_PROMPT = """You are an expert evaluator for an insurance customer support chatbot.

Rate how well the generated answer addresses the customer's question.

Question: {question}

Expected answer (reference): {expected}

Generated answer: {answer}

Rate the relevance on a scale of 0.0 to 1.0:
- 1.0: Perfectly answers the question with correct, complete information
- 0.8: Mostly correct, minor details missing
- 0.6: Partially correct, some relevant information but incomplete
- 0.4: Tangentially relevant, misses key points
- 0.2: Mostly irrelevant or incorrect
- 0.0: Completely wrong or off-topic

Respond with ONLY a JSON object: {{"score": <float>, "reason": "<brief explanation>"}}"""

_QUALITY_PROMPT = """You are evaluating the conversational quality of a customer support chatbot response.

Question: {question}
Answer: {answer}
Language: {language}

Rate the conversational quality on a scale of 0.0 to 1.0 based on:
- Clarity: Is the answer easy to understand?
- Tone: Is it professional and helpful?
- Structure: Is it well-organized (bullets, sections)?
- Completeness: Does it guide the customer on next steps if needed?
- Language: Does it respond in the correct language?

Respond with ONLY a JSON object: {{"score": <float>, "reason": "<brief explanation>"}}"""


def _parse_score(response: str) -> float:
    """Extract score from LLM response."""
    match = re.search(r'"score"\s*:\s*([\d.]+)', response)
    if match:
        return max(0.0, min(1.0, float(match.group(1))))
    # Fallback: try to find any float
    match = re.search(r"\b(0\.\d+|1\.0|0|1)\b", response)
    if match:
        return float(match.group(1))
    logger.warning(f"Could not parse score from: {response[:100]}")
    return 0.0


def score_relevance(
    question: str,
    expected_answer: str,
    generated_answer: str,
    client: ClaudeClient | None = None,
) -> float:
    """Score answer relevance using Claude as judge.

    Returns:
        Float between 0.0 and 1.0.
    """
    if not generated_answer or not generated_answer.strip():
        return 0.0

    client = client or ClaudeClient()
    prompt = _RELEVANCE_PROMPT.format(
        question=question,
        expected=expected_answer or "(no reference answer provided)",
        answer=generated_answer,
    )

    try:
        response = client.generate(prompt, temperature=0.0, max_tokens=256)
        return _parse_score(response)
    except Exception as e:
        logger.error(f"Relevance scoring failed: {e}")
        return 0.0


def score_conversational_quality(
    question: str,
    answer: str,
    language: str = "he",
    client: ClaudeClient | None = None,
) -> float:
    """Score conversational quality using Claude as judge.

    Returns:
        Float between 0.0 and 1.0.
    """
    if not answer or not answer.strip():
        return 0.0

    client = client or ClaudeClient()
    prompt = _QUALITY_PROMPT.format(
        question=question,
        answer=answer,
        language=language,
    )

    try:
        response = client.generate(prompt, temperature=0.0, max_tokens=256)
        return _parse_score(response)
    except Exception as e:
        logger.error(f"Quality scoring failed: {e}")
        return 0.0
