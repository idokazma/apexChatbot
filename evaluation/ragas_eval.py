"""RAGAS-based evaluation harness for the RAG pipeline."""

import json
import time
from pathlib import Path

from loguru import logger

from evaluation.citation_scorer import score_citations
from evaluation.metrics import EvalResult, aggregate_scores


def run_evaluation(
    agent,
    questions_path: Path,
    output_dir: Path,
) -> dict:
    """Run full evaluation on a question set.

    Args:
        agent: Compiled LangGraph agent.
        questions_path: Path to JSON file with test questions.
        output_dir: Directory to save evaluation reports.

    Returns:
        Aggregated evaluation scores.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load questions
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    logger.info(f"Evaluating {len(questions)} questions...")

    results: list[EvalResult] = []

    for i, q in enumerate(questions):
        query = q["question"]
        expected = q.get("expected_answer", "")
        domain = q.get("domain", "unknown")

        logger.info(f"[{i+1}/{len(questions)}] {query[:60]}...")

        # Run agent
        start = time.time()
        agent_input = {
            "query": query,
            "messages": [],
            "rewritten_query": "",
            "detected_domains": [],
            "detected_language": "he",
            "retrieved_documents": [],
            "graded_documents": [],
            "generation": "",
            "citations": [],
            "is_grounded": False,
            "retry_count": 0,
            "should_fallback": False,
        }

        output = agent.invoke(agent_input)
        latency = time.time() - start

        answer = output.get("generation", "")
        citations = output.get("citations", [])
        graded_docs = output.get("graded_documents", [])

        # Score citations
        cit_scores = score_citations(answer, citations, graded_docs)

        # Efficiency score (target: <5s = 1.0, >15s = 0.0)
        efficiency = max(0.0, min(1.0, 1.0 - (latency - 5.0) / 10.0))

        result = EvalResult(
            question=query,
            expected_answer=expected,
            generated_answer=answer,
            citations=citations,
            domain=domain,
            citation_accuracy=cit_scores["score"],
            efficiency=efficiency,
            # Relevance and conversational quality need manual or LLM-based scoring
            relevance=0.0,  # TODO: Add RAGAS answer_relevancy
            conversational_quality=0.0,  # TODO: Add coherence metric
        )
        results.append(result)

    # Aggregate
    scores = aggregate_scores(results)

    # Save report
    report_path = output_dir / "eval_report.json"
    report_path.write_text(json.dumps(scores, ensure_ascii=False, indent=2))
    logger.info(f"Evaluation complete. Report saved to {report_path}")
    logger.info(f"Average weighted score: {scores.get('avg_weighted_score', 0):.3f}")

    return scores
