"""Evaluation harness: runs agent on test questions and scores with LLM-as-judge."""

import json
import time
from pathlib import Path

from loguru import logger

from evaluation.citation_scorer import score_citations
from evaluation.keyword_scorer import score_keywords
from evaluation.llm_judge import score_conversational_quality, score_relevance
from evaluation.metrics import EvalResult, aggregate_scores
from llm.claude_client import ClaudeClient


def run_evaluation(
    agent,
    questions_path: Path,
    output_dir: Path,
    use_llm_judge: bool = True,
) -> dict:
    """Run full evaluation on a question set.

    Args:
        agent: Compiled LangGraph agent.
        questions_path: Path to JSON file with test questions.
        output_dir: Directory to save evaluation reports.
        use_llm_judge: Whether to use Claude for relevance/quality scoring.

    Returns:
        Aggregated evaluation scores.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load questions
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    logger.info(f"Evaluating {len(questions)} questions...")

    # Initialize LLM judge if enabled
    judge = ClaudeClient() if use_llm_judge else None

    results: list[EvalResult] = []
    details: list[dict] = []

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
            "quality_action": "",
            "quality_reasoning": "",
            "reasoning_trace": [],
        }

        output = agent.invoke(agent_input)
        latency = time.time() - start

        answer = output.get("generation", "")
        citations = output.get("citations", [])
        graded_docs = output.get("graded_documents", [])
        detected_language = output.get("detected_language", "he")

        # Score citations
        cit_scores = score_citations(answer, citations, graded_docs)

        # Keyword scoring (if annotations exist)
        kw_score = 0.0
        if q.get("required_keywords"):
            kw_result = score_keywords(
                answer, q.get("required_keywords", []), q.get("forbidden_keywords", [])
            )
            kw_score = kw_result.score

        # Efficiency score (target: <5s = 1.0, >15s = 0.0)
        efficiency = max(0.0, min(1.0, 1.0 - (latency - 5.0) / 10.0))

        # LLM-as-judge scoring
        relevance = 0.0
        quality = 0.0
        if use_llm_judge and judge:
            relevance = score_relevance(query, expected, answer, client=judge)
            quality = score_conversational_quality(
                query, answer, language=detected_language, client=judge,
            )

        result = EvalResult(
            question=query,
            expected_answer=expected,
            generated_answer=answer,
            citations=citations,
            domain=domain,
            citation_accuracy=cit_scores["score"],
            efficiency=efficiency,
            relevance=relevance,
            conversational_quality=quality,
        )
        results.append(result)

        # Detailed per-question log
        details.append({
            "question": query,
            "domain": domain,
            "answer": answer[:500],
            "latency_s": round(latency, 2),
            "relevance": round(relevance, 3),
            "citation_accuracy": round(cit_scores["score"], 3),
            "efficiency": round(efficiency, 3),
            "conversational_quality": round(quality, 3),
            "weighted_score": round(result.weighted_score, 3),
            "is_grounded": output.get("is_grounded", False),
            "num_citations": len(citations),
            "keyword_score": round(kw_score, 3),
            "num_graded_docs": len(graded_docs),
        })

        logger.info(
            f"  → relevance={relevance:.2f} citation={cit_scores['score']:.2f} "
            f"quality={quality:.2f} latency={latency:.1f}s"
        )

    # Aggregate
    scores = aggregate_scores(results)

    # Save reports
    report_path = output_dir / "eval_report.json"
    report_path.write_text(json.dumps(scores, ensure_ascii=False, indent=2))

    details_path = output_dir / "eval_details.json"
    details_path.write_text(json.dumps(details, ensure_ascii=False, indent=2))

    logger.info(f"Evaluation complete. Reports saved to {output_dir}")
    logger.info(f"Average weighted score: {scores.get('avg_weighted_score', 0):.3f}")

    return scores
