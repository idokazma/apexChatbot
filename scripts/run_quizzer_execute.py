"""Execute quizzer using previously generated questions with longer timeout."""

import json
import sys
import time
from pathlib import Path

from loguru import logger

from quizzer.answer_scorer import QuizScore, score_answer
from quizzer.api_client import ChatbotAPIClient
from quizzer.question_generator import GeneratedQuestion
from quizzer.question_types import QuestionType
from quizzer.runner import QuizRunConfig, QuizRunResult, _avg, _save_intermediate
from quizzer.report_generator import generate_report
from llm.ollama_client import OllamaClient


def main():
    questions_path = Path("quizzer/reports/generated_questions.json")
    if not questions_path.exists():
        logger.error(f"No generated questions found at {questions_path}")
        sys.exit(1)

    raw = json.loads(questions_path.read_text(encoding="utf-8"))
    questions = []
    for q in raw:
        if "source_doc_titles" in q and "source_documents" not in q:
            q["source_documents"] = [{"title": t} for t in q.pop("source_doc_titles")]
        if isinstance(q.get("question_type"), str):
            q["question_type"] = QuestionType(q["question_type"])
        questions.append(GeneratedQuestion(**q))
    logger.info(f"Loaded {len(questions)} pre-generated questions")

    config = QuizRunConfig(
        num_questions=len(questions),
        api_base_url="http://localhost:8000",
        api_timeout=180.0,
        output_dir="quizzer/reports",
        save_intermediate=True,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = QuizRunResult(config=config)
    result.questions_generated = len(questions)

    # API client with no retries and long timeout
    api = ChatbotAPIClient(base_url=config.api_base_url, timeout=180.0, max_retries=0)
    if not api.health_check():
        logger.error("API not healthy")
        sys.exit(1)

    # Use Ollama for scoring (Claude credits exhausted)
    scorer_llm = OllamaClient(model="gemma3:12b")
    logger.info(f"Using Ollama (gemma3:12b) for answer scoring")
    query_start = time.time()

    for i, question in enumerate(questions):
        logger.info(f"[{i+1}/{len(questions)}] Asking: {question.question[:80]}...")
        api_response = api.ask(question.question, language=question.language)
        result.questions_asked += 1

        if not api_response.get("success"):
            result.api_failures += 1
            logger.warning(f"  -> FAILED (timeout or error)")
        else:
            logger.info(f"  -> OK ({api_response['latency_s']:.1f}s, domain={api_response.get('domain')})")

        score = score_answer(question, api_response, scorer_llm)
        result.scores.append(score)
        logger.info(f"  -> Score: {score.overall_score:.3f}")

        if (i + 1) % 10 == 0:
            avg_score = _avg([s.overall_score for s in result.scores])
            logger.info(
                f"=== Progress: [{i+1}/{len(questions)}] "
                f"avg_score={avg_score:.3f} "
                f"failures={result.api_failures} ==="
            )

    result.querying_time_s = time.time() - query_start
    api.close()

    # Save results
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

    # Generate HTML report
    from quizzer.report_generator import _build_html, _build_cross_tab
    data = result.to_dict()
    details = data.get("details", [])
    cross_tab = _build_cross_tab(details)
    scored = sorted(details, key=lambda d: d.get("overall_score", 0))
    best_10 = list(reversed(scored[-10:]))
    worst_10 = scored[:10]
    html = _build_html(
        data.get("summary", {}), data.get("overall_metrics", {}),
        data.get("by_question_type", {}), data.get("by_domain", {}),
        data.get("by_difficulty", {}), data.get("by_language", {}),
        cross_tab, best_10, worst_10, "", details,
    )
    report_path = output_dir / f"quiz_report_{ts}.html"
    report_path.write_text(html, encoding="utf-8")
    latest_report = output_dir / "quiz_report.html"
    latest_report.write_text(html, encoding="utf-8")

    if result.scores:
        avg = sum(s.overall_score for s in result.scores) / len(result.scores)
        logger.info(f"Quiz complete: avg_score={avg:.3f}")
        logger.info(f"Report: {latest_report}")
    else:
        logger.error("No scores collected.")
        sys.exit(1)


if __name__ == "__main__":
    main()
