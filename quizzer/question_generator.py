"""Generate realistic customer questions from sampled documents using Claude."""

import json
import random

from loguru import logger

from llm.gemini_client import GeminiClient
from quizzer.prompts import QUESTION_GENERATION_PROMPT
from quizzer.question_types import (
    QUESTION_TYPE_DESCRIPTIONS,
    QUESTION_TYPE_WEIGHTS,
    GeneratedQuestion,
    QuestionType,
)


def _pick_question_type() -> QuestionType:
    """Randomly select a question type based on configured weights."""
    types = list(QUESTION_TYPE_WEIGHTS.keys())
    weights = [QUESTION_TYPE_WEIGHTS[t] for t in types]
    return random.choices(types, weights=weights, k=1)[0]


def _pick_difficulty() -> str:
    """Randomly select question difficulty."""
    return random.choices(
        ["easy", "medium", "hard"],
        weights=[0.4, 0.4, 0.2],
        k=1,
    )[0]


def _pick_language() -> str:
    """Randomly select question language (Hebrew-heavy, some English)."""
    return random.choices(["he", "en"], weights=[0.8, 0.2], k=1)[0]


def _format_docs_for_prompt(docs: list[dict]) -> str:
    """Format document chunks into a readable string for the prompt."""
    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(
            f"--- Document {i} ---\n"
            f"Domain: {doc.get('domain', 'unknown')}\n"
            f"Title: {doc.get('source_doc_title', 'N/A')}\n"
            f"Section: {doc.get('section_path', 'N/A')}\n"
            f"Content:\n{doc.get('content', '')}\n"
        )
    return "\n".join(parts)


def generate_question(
    docs: list[dict],
    client: GeminiClient,
    question_type: QuestionType | None = None,
    language: str | None = None,
    difficulty: str | None = None,
) -> GeneratedQuestion | None:
    """Generate a single question from a group of documents.

    Args:
        docs: List of document chunk dicts to base the question on.
        client: GeminiClient for LLM calls.
        question_type: Force a specific question type, or None for random.
        language: Force a language, or None for random.
        difficulty: Force difficulty, or None for random.

    Returns:
        GeneratedQuestion or None if generation failed.
    """
    qtype = question_type or _pick_question_type()
    lang = language or _pick_language()
    diff = difficulty or _pick_difficulty()
    domain = docs[0].get("domain", "unknown") if docs else "unknown"

    lang_name = "Hebrew" if lang == "he" else "English"

    prompt = QUESTION_GENERATION_PROMPT.format(
        documents=_format_docs_for_prompt(docs),
        question_type=qtype.value,
        type_description=QUESTION_TYPE_DESCRIPTIONS[qtype],
        language=lang_name,
        difficulty=diff,
    )

    try:
        response = client.generate(prompt, temperature=0.7, max_tokens=1024)
        data = json.loads(response)

        if not data.get("answerable", False) or not data.get("question"):
            logger.debug(f"Unanswerable question for {qtype.value}/{domain}, skipping")
            return None

        return GeneratedQuestion(
            question=data["question"],
            question_type=qtype,
            domain=domain,
            source_documents=docs,
            expected_answer=data.get("ground_truth_answer", ""),
            expected_answer_hints=data.get("expected_answer_hints", ""),
            language=lang,
            difficulty=diff,
        )

    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Failed to parse question generation response: {e}")
        return None
    except Exception as e:
        logger.error(f"Question generation failed: {e}")
        return None


def generate_questions_batch(
    doc_groups: list[list[dict]],
    client: GeminiClient,
    target_count: int = 1000,
) -> list[GeneratedQuestion]:
    """Generate a batch of questions from document groups.

    Retries with different groups if generation fails, until target_count
    questions are generated or groups are exhausted.

    Args:
        doc_groups: List of document groups from document_sampler.
        client: GeminiClient for LLM calls.
        target_count: Target number of questions to generate.

    Returns:
        List of GeneratedQuestion objects.
    """
    questions: list[GeneratedQuestion] = []
    attempts = 0
    max_attempts = target_count * 2  # Allow for some failures
    group_idx = 0

    logger.info(f"Generating {target_count} questions from {len(doc_groups)} doc groups...")

    while len(questions) < target_count and attempts < max_attempts:
        # Cycle through doc groups, wrapping around if needed
        docs = doc_groups[group_idx % len(doc_groups)]
        group_idx += 1

        q = generate_question(docs, client)
        attempts += 1

        if q is not None:
            questions.append(q)
            if len(questions) % 50 == 0:
                logger.info(
                    f"Generated {len(questions)}/{target_count} questions "
                    f"({attempts} attempts)"
                )

    logger.info(
        f"Question generation complete: {len(questions)} questions "
        f"in {attempts} attempts"
    )
    return questions
