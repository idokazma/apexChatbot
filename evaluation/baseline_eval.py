"""GPT baseline evaluation: run same questions through GPT-4o / GPT-5 without RAG."""

import json
import time
from pathlib import Path

from loguru import logger
from openai import OpenAI

from config.settings import settings
from evaluation.citation_scorer import score_citations
from evaluation.llm_judge import score_conversational_quality, score_relevance
from evaluation.metrics import EvalResult, aggregate_scores
from llm.claude_client import ClaudeClient

BASELINE_SYSTEM_PROMPT = """You are a professional customer support assistant for Harel Insurance (הראל ביטוח), \
Israel's largest insurance and financial services group.

Answer customer questions about insurance policies accurately and helpfully.
Respond in the same language the customer uses (Hebrew or English).
If you're not sure about specific policy details, say so clearly.

Insurance domains: Car, Life, Travel, Health, Dental, Mortgage, Business, Apartment."""


def run_baseline(
    questions_path: Path,
    output_dir: Path,
    model: str = "gpt-4o",
) -> dict:
    """Run baseline evaluation using GPT without RAG.

    Args:
        questions_path: Path to JSON file with test questions.
        output_dir: Directory to save results.
        model: OpenAI model name (e.g., "gpt-4o", "gpt-5").

    Returns:
        Aggregated evaluation scores.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=settings.openai_api_key)
    judge = ClaudeClient()

    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    logger.info(f"Running baseline ({model}) on {len(questions)} questions...")

    results: list[EvalResult] = []
    details: list[dict] = []

    for i, q in enumerate(questions):
        query = q["question"]
        expected = q.get("expected_answer", "")
        domain = q.get("domain", "unknown")

        logger.info(f"[{i+1}/{len(questions)}] {query[:60]}...")

        start = time.time()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0.1,
                max_tokens=2048,
            )
            answer = response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"GPT call failed: {e}")
            answer = ""

        latency = time.time() - start

        # Baseline has no citations (that's the point)
        cit_scores = score_citations(answer, [], [])
        efficiency = max(0.0, min(1.0, 1.0 - (latency - 5.0) / 10.0))

        relevance = score_relevance(query, expected, answer, client=judge)
        quality = score_conversational_quality(query, answer, client=judge)

        result = EvalResult(
            question=query,
            expected_answer=expected,
            generated_answer=answer,
            citations=[],
            domain=domain,
            citation_accuracy=cit_scores["score"],
            efficiency=efficiency,
            relevance=relevance,
            conversational_quality=quality,
        )
        results.append(result)

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
        })

        logger.info(
            f"  → relevance={relevance:.2f} citation={cit_scores['score']:.2f} "
            f"quality={quality:.2f} latency={latency:.1f}s"
        )

    scores = aggregate_scores(results)
    scores["model"] = model

    report_path = output_dir / f"baseline_{model.replace('-', '_')}_report.json"
    report_path.write_text(json.dumps(scores, ensure_ascii=False, indent=2))

    details_path = output_dir / f"baseline_{model.replace('-', '_')}_details.json"
    details_path.write_text(json.dumps(details, ensure_ascii=False, indent=2))

    logger.info(f"Baseline ({model}) complete. Average weighted: {scores.get('avg_weighted_score', 0):.3f}")

    return scores
