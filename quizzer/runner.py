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
        "avg_type_accuracy": _avg([s.type_accuracy for s in scores]),
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


def prepare_questions(
    config: QuizRunConfig,
    progress_callback: callable | None = None,
) -> list[GeneratedQuestion]:
    """Phase 1: Generate all questions upfront without querying the system.

    Args:
        config: Quiz run configuration.
        progress_callback: Optional callback(phase: str, detail: str) for progress updates.

    Returns:
        List of GeneratedQuestion ready for execution.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _notify(phase: str, detail: str = "") -> None:
        if progress_callback:
            progress_callback(phase, detail)

    # Step 1: Connect to vector store
    _notify("Connecting to vector store")
    store = VectorStoreClient()
    store.connect()
    doc_count = store.get_count()
    logger.info(f"[PREPARE] Vector store has {doc_count} documents")

    if doc_count == 0:
        store.disconnect()
        raise RuntimeError("No documents in vector store. Run the data pipeline first.")

    # Step 2: Sample document groups
    _notify("Sampling documents")
    n_groups = int(config.num_questions * 1.5)
    doc_groups = sample_document_groups(
        store,
        n_groups=n_groups,
        docs_per_group=config.docs_per_question,
    )

    # Step 3: Generate questions
    _notify("Generating questions")
    claude = ClaudeClient()
    questions = generate_questions_batch(doc_groups, claude, target_count=config.num_questions)
    logger.info(f"[PREPARE] Generated {len(questions)} questions")

    # Save questions for reproducibility
    if config.save_intermediate:
        _save_questions(questions, output_dir / "generated_questions.json")

    store.disconnect()
    _notify("Ready", f"{len(questions)} questions generated")
    return questions


def execute_questions(
    questions: list[GeneratedQuestion],
    config: QuizRunConfig,
    per_question_callback: callable | None = None,
) -> QuizRunResult:
    """Phase 2: Query the system for each prepared question and score it.

    Args:
        questions: Pre-generated questions from prepare_questions().
        config: Quiz run configuration.
        per_question_callback: Optional callback(index: int, score: QuizScore) called after each question.

    Returns:
        QuizRunResult with all scores and metadata.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = QuizRunResult(config=config)
    result.questions_generated = len(questions)

    # Check API health
    api = ChatbotAPIClient(base_url=config.api_base_url, timeout=config.api_timeout)
    if not api.health_check():
        api.close()
        raise RuntimeError(f"API at {config.api_base_url} is not healthy. Start the server with 'make serve' first.")

    claude = ClaudeClient()
    query_start = time.time()

    for i, question in enumerate(questions):
        api_response = api.ask(question.question, language=question.language)
        result.questions_asked += 1

        if not api_response.get("success"):
            result.api_failures += 1

        score = score_answer(question, api_response, claude)
        result.scores.append(score)

        if per_question_callback:
            per_question_callback(i, score)

        if (i + 1) % 25 == 0:
            avg_score = _avg([s.overall_score for s in result.scores])
            logger.info(
                f"[EXECUTE] [{i + 1}/{len(questions)}] "
                f"avg_score={avg_score:.3f} "
                f"failures={result.api_failures} "
                f"latency={score.latency_s:.1f}s"
            )

        if config.save_intermediate and (i + 1) % 100 == 0:
            _save_intermediate(result, output_dir / "intermediate_results.json")

    result.querying_time_s = time.time() - query_start
    api.close()

    # Generate report
    ts = time.strftime("%Y%m%d_%H%M%S")
    raw_path = output_dir / f"quiz_results_{ts}.json"
    raw_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    latest_path = output_dir / "quiz_results.json"
    latest_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_path = output_dir / f"quiz_report_{ts}.html"
    generate_report(result, report_path, claude)
    latest_html = output_dir / "quiz_report.html"
    generate_report(result, latest_html, claude)

    logger.info(f"[EXECUTE] Complete: {len(result.scores)} scores, avg={_avg([s.overall_score for s in result.scores]):.3f}")
    return result


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


def save_quiz_set(questions: list[GeneratedQuestion], output_dir: str | Path) -> Path:
    """Save a full quiz set with source documents and ground truth answers.

    Args:
        questions: Generated questions with source_documents.
        output_dir: Directory to save the quiz set file.

    Returns:
        Path to the saved quiz set file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    name = f"quiz_set_{ts}"
    max_content_len = 1000

    quiz_questions = []
    for q in questions:
        # Build source docs with truncated content
        source_docs = []
        for d in q.source_documents:
            content = d.get("content", "")
            source_docs.append({
                "source_url": d.get("source_url", ""),
                "source_doc_title": d.get("source_doc_title", ""),
                "domain": d.get("domain", ""),
                "content": content[:max_content_len] if content else "",
            })

        # Build ground truth answer from source documents and hints
        gt_parts = []
        if q.expected_answer_hints:
            gt_parts.append(q.expected_answer_hints)
        gt_parts.append("\n\nReferences:")
        for i, d in enumerate(source_docs, 1):
            title = d.get("source_doc_title", "Unknown")
            url = d.get("source_url", "")
            ref = f"[{i}] {title}"
            if url:
                ref += f" ({url})"
            gt_parts.append(ref)

        quiz_questions.append({
            "question": q.question,
            "question_type": q.question_type.value,
            "domain": q.domain,
            "language": q.language,
            "difficulty": q.difficulty,
            "expected_answer_hints": q.expected_answer_hints,
            "ground_truth_answer": "\n".join(gt_parts),
            "source_documents": source_docs,
        })

    quiz_set = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "num_questions": len(quiz_questions),
        "name": name,
        "questions": quiz_questions,
    }

    path = output_dir / f"{name}.json"
    path.write_text(json.dumps(quiz_set, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[QUIZ SET] Saved {len(quiz_questions)} questions to {path}")
    return path


def load_quiz_set(path: str | Path) -> list[GeneratedQuestion]:
    """Load a saved quiz set and reconstruct GeneratedQuestion objects.

    Args:
        path: Path to the quiz set JSON file.

    Returns:
        List of GeneratedQuestion objects ready for execution.
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    questions_data = data.get("questions", [])

    questions = []
    for q in questions_data:
        questions.append(GeneratedQuestion(
            question=q["question"],
            question_type=QuestionType(q["question_type"]),
            domain=q["domain"],
            source_documents=q.get("source_documents", []),
            expected_answer_hints=q.get("expected_answer_hints", ""),
            language=q.get("language", "he"),
            difficulty=q.get("difficulty", "medium"),
        ))

    logger.info(f"[QUIZ SET] Loaded {len(questions)} questions from {path}")
    return questions


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
