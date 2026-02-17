"""Orchestrates the full quiz run: sample docs, generate questions, query API, score answers."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from data_pipeline.store.vector_store import VectorStoreClient
from llm.claude_client import ClaudeClient
from quizzer.answer_scorer import QuizScore, score_answer
from quizzer.api_client import ChatbotAPIClient
from quizzer.document_sampler import sample_document_groups
from quizzer.question_generator import generate_questions_batch
from quizzer.question_types import GeneratedQuestion, QuestionType
from quizzer.report_generator import generate_report


@dataclass
class QuizRunConfig:
    """Configuration for a quiz run."""

    num_questions: int = 1000
    docs_per_question: int = 2
    api_base_url: str = "http://localhost:8000"
    api_timeout: float = 60.0
    output_dir: str = "quizzer/reports"
    save_intermediate: bool = True


@dataclass
class QuizRunResult:
    """Complete results from a quiz run."""

    config: QuizRunConfig
    scores: list[QuizScore] = field(default_factory=list)
    questions_generated: int = 0
    questions_asked: int = 0
    api_failures: int = 0
    total_time_s: float = 0.0
    generation_time_s: float = 0.0
    querying_time_s: float = 0.0

    def to_dict(self) -> dict:
        """Serialize the full result for JSON export."""
        return {
            "summary": {
                "num_questions": self.config.num_questions,
                "questions_generated": self.questions_generated,
                "questions_asked": self.questions_asked,
                "api_failures": self.api_failures,
                "total_time_s": round(self.total_time_s, 1),
                "generation_time_s": round(self.generation_time_s, 1),
                "querying_time_s": round(self.querying_time_s, 1),
            },
            "overall_metrics": _compute_overall_metrics(self.scores),
            "by_question_type": _compute_by_question_type(self.scores),
            "by_domain": _compute_by_domain(self.scores),
            "by_difficulty": _compute_by_difficulty(self.scores),
            "by_language": _compute_by_language(self.scores),
            "details": [s.to_dict() for s in self.scores],
        }


def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def _compute_group_metrics(scores: list[QuizScore]) -> dict:
    """Compute aggregate metrics for a group of scores."""
    if not scores:
        return {}
    return {
        "count": len(scores),
        "avg_overall": _avg([s.overall_score for s in scores]),
        "avg_correctness": _avg([s.correctness for s in scores]),
        "avg_completeness": _avg([s.completeness for s in scores]),
        "avg_relevance": _avg([s.relevance for s in scores]),
        "avg_citation_quality": _avg([s.citation_quality for s in scores]),
        "avg_citation_f1": _avg([s.citation_f1 for s in scores]),
        "avg_tone": _avg([s.tone for s in scores]),
        "avg_efficiency": _avg([s.efficiency for s in scores]),
        "avg_latency_s": _avg([s.latency_s for s in scores]),
        "domain_match_rate": _avg([1.0 if s.domain_match else 0.0 for s in scores]),
        "failure_rate": _avg([0.0 if s.success else 1.0 for s in scores]),
    }


def _compute_overall_metrics(scores: list[QuizScore]) -> dict:
    return _compute_group_metrics(scores)


def _compute_by_question_type(scores: list[QuizScore]) -> dict:
    groups: dict[str, list[QuizScore]] = {}
    for s in scores:
        groups.setdefault(s.question_type.value, []).append(s)
    return {k: _compute_group_metrics(v) for k, v in sorted(groups.items())}


def _compute_by_domain(scores: list[QuizScore]) -> dict:
    groups: dict[str, list[QuizScore]] = {}
    for s in scores:
        groups.setdefault(s.domain, []).append(s)
    return {k: _compute_group_metrics(v) for k, v in sorted(groups.items())}


def _compute_by_difficulty(scores: list[QuizScore]) -> dict:
    groups: dict[str, list[QuizScore]] = {}
    for s in scores:
        groups.setdefault(s.difficulty, []).append(s)
    return {k: _compute_group_metrics(v) for k, v in sorted(groups.items())}


def _compute_by_language(scores: list[QuizScore]) -> dict:
    groups: dict[str, list[QuizScore]] = {}
    for s in scores:
        groups.setdefault(s.language, []).append(s)
    return {k: _compute_group_metrics(v) for k, v in sorted(groups.items())}


def run_quiz(config: QuizRunConfig | None = None) -> QuizRunResult:
    """Execute a full quiz run.

    Steps:
        1. Connect to ChromaDB and sample document groups.
        2. Generate questions using Claude.
        3. Send each question to the chatbot API.
        4. Score each answer.
        5. Generate a visual report.

    Args:
        config: Quiz run configuration. Uses defaults if None.

    Returns:
        QuizRunResult with all scores and metadata.
    """
    config = config or QuizRunConfig()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = QuizRunResult(config=config)
    run_start = time.time()

    # --- Step 1: Connect to vector store ---
    logger.info("Connecting to ChromaDB...")
    store = VectorStoreClient()
    store.connect()
    doc_count = store.get_count()
    logger.info(f"Vector store has {doc_count} documents")

    if doc_count == 0:
        logger.error("No documents in vector store. Run the data pipeline first.")
        return result

    # --- Step 2: Sample document groups ---
    logger.info("Sampling document groups...")
    # Oversample groups to account for question generation failures
    n_groups = int(config.num_questions * 1.5)
    doc_groups = sample_document_groups(
        store,
        n_groups=n_groups,
        docs_per_group=config.docs_per_question,
    )

    # --- Step 3: Generate questions ---
    logger.info(f"Generating {config.num_questions} questions...")
    gen_start = time.time()
    claude = ClaudeClient()
    questions = generate_questions_batch(
        doc_groups, claude, target_count=config.num_questions,
    )
    result.generation_time_s = time.time() - gen_start
    result.questions_generated = len(questions)
    logger.info(f"Generated {len(questions)} questions in {result.generation_time_s:.0f}s")

    # Save generated questions for reproducibility
    if config.save_intermediate:
        _save_questions(questions, output_dir / "generated_questions.json")

    # --- Step 4: Query the API ---
    logger.info("Checking API health...")
    api = ChatbotAPIClient(
        base_url=config.api_base_url,
        timeout=config.api_timeout,
    )

    if not api.health_check():
        logger.error(
            f"API at {config.api_base_url} is not healthy. "
            "Start the server with 'make serve' first."
        )
        store.disconnect()
        return result

    logger.info(f"Sending {len(questions)} questions to the chatbot API...")
    query_start = time.time()

    for i, question in enumerate(questions):
        # Ask the chatbot
        api_response = api.ask(question.question, language=question.language)
        result.questions_asked += 1

        if not api_response.get("success"):
            result.api_failures += 1

        # Score the answer
        score = score_answer(question, api_response, claude)
        result.scores.append(score)

        # Progress logging
        if (i + 1) % 25 == 0:
            avg_score = _avg([s.overall_score for s in result.scores])
            logger.info(
                f"[{i + 1}/{len(questions)}] "
                f"avg_score={avg_score:.3f} "
                f"failures={result.api_failures} "
                f"latency={score.latency_s:.1f}s"
            )

        # Save intermediate results periodically
        if config.save_intermediate and (i + 1) % 100 == 0:
            _save_intermediate(result, output_dir / "intermediate_results.json")

    result.querying_time_s = time.time() - query_start
    api.close()

    # --- Step 5: Generate report ---
    result.total_time_s = time.time() - run_start
    logger.info(
        f"Quiz complete: {len(result.scores)} questions scored in "
        f"{result.total_time_s:.0f}s"
    )

    # Save raw results
    raw_path = output_dir / "quiz_results.json"
    raw_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"Raw results saved to {raw_path}")

    # Generate visual HTML report
    report_path = output_dir / "quiz_report.html"
    generate_report(result, report_path, claude)
    logger.info(f"Visual report saved to {report_path}")

    store.disconnect()
    return result


def _save_questions(questions: list[GeneratedQuestion], path: Path) -> None:
    """Save generated questions to JSON for reproducibility."""
    data = [
        {
            "question": q.question,
            "question_type": q.question_type.value,
            "domain": q.domain,
            "language": q.language,
            "difficulty": q.difficulty,
            "expected_answer_hints": q.expected_answer_hints,
            "source_doc_titles": [
                d.get("source_doc_title", "") for d in q.source_documents
            ],
        }
        for q in questions
    ]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.debug(f"Saved {len(data)} questions to {path}")


def _save_intermediate(result: QuizRunResult, path: Path) -> None:
    """Save intermediate results for crash recovery."""
    data = {
        "questions_asked": result.questions_asked,
        "api_failures": result.api_failures,
        "scores_count": len(result.scores),
        "avg_overall": _avg([s.overall_score for s in result.scores]),
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
