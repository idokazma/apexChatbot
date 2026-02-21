"""Ground Truth Evaluator — sends curated Q&A pairs to /chat and scores the results."""

import json
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

from loguru import logger

from llm.gemini_client import GeminiClient
from quizzer.api_client import ChatbotAPIClient

# ---------------------------------------------------------------------------
# Hebrew domain key → internal domain name
# ---------------------------------------------------------------------------
DOMAIN_MAP: dict[str, str] = {
    "דירה": "apartment",
    "רכב": "car",
    "חיים": "life",
    "נסיעות": "travel",
    "בריאות": "health",
    "שיניים": "dental",
    "משכנתא": "mortgage",
    "עסקים": "business",
}

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GTQuestion:
    question: str
    expected_answer: str
    expected_file: str
    expected_page: int
    domain: str


@dataclass
class GTQuestionResult:
    question: str
    expected_answer: str
    expected_file: str
    expected_page: int
    domain: str
    system_answer: str = ""
    system_citations: list[dict] = field(default_factory=list)
    system_domain: str | None = None
    latency_s: float = 0.0
    api_success: bool = False
    answer_correctness: float = 0.0
    answer_reasoning: str = ""
    file_match: bool = False
    page_match: bool = False
    domain_match: bool = False
    matched_citation_file: str | None = None

    @property
    def pass_all(self) -> bool:
        return self.answer_correctness >= 0.5 and self.file_match and self.page_match


@dataclass
class GTEvalResult:
    results: list[GTQuestionResult] = field(default_factory=list)
    total_time_s: float = 0.0

    def to_dict(self) -> dict:
        passed = [r for r in self.results if r.pass_all]
        failed = [r for r in self.results if not r.pass_all]
        total = len(self.results)

        correctness_vals = [r.answer_correctness for r in self.results]
        latency_vals = [r.latency_s for r in self.results]

        summary = {
            "total_questions": total,
            "passed": len(passed),
            "failed": len(failed),
            "pass_rate": round(len(passed) / total, 4) if total else 0,
            "avg_answer_correctness": round(sum(correctness_vals) / total, 4) if total else 0,
            "file_match_rate": round(sum(r.file_match for r in self.results) / total, 4) if total else 0,
            "page_match_rate": round(sum(r.page_match for r in self.results) / total, 4) if total else 0,
            "domain_match_rate": round(sum(r.domain_match for r in self.results) / total, 4) if total else 0,
            "avg_latency_s": round(sum(latency_vals) / total, 2) if total else 0,
            "total_time_s": round(self.total_time_s, 2),
        }

        # Per-domain breakdown
        domains: dict[str, list[GTQuestionResult]] = {}
        for r in self.results:
            domains.setdefault(r.domain, []).append(r)

        by_domain = {}
        for domain, items in sorted(domains.items()):
            n = len(items)
            dp = [r for r in items if r.pass_all]
            by_domain[domain] = {
                "count": n,
                "passed": len(dp),
                "pass_rate": round(len(dp) / n, 4),
                "avg_answer_correctness": round(sum(r.answer_correctness for r in items) / n, 4),
                "file_match_rate": round(sum(r.file_match for r in items) / n, 4),
                "page_match_rate": round(sum(r.page_match for r in items) / n, 4),
            }

        details = []
        for r in self.results:
            details.append({
                "question": r.question,
                "domain": r.domain,
                "expected_answer": r.expected_answer,
                "system_answer": r.system_answer,
                "expected_file": r.expected_file,
                "matched_citation_file": r.matched_citation_file,
                "expected_page": r.expected_page,
                "answer_correctness": r.answer_correctness,
                "answer_reasoning": r.answer_reasoning,
                "file_match": r.file_match,
                "page_match": r.page_match,
                "domain_match": r.domain_match,
                "pass_all": r.pass_all,
                "latency_s": r.latency_s,
                "api_success": r.api_success,
            })

        return {"summary": summary, "by_domain": by_domain, "details": details}


# ---------------------------------------------------------------------------
# Ground truth loader
# ---------------------------------------------------------------------------

def load_ground_truth(path: str | Path) -> list[GTQuestion]:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    questions: list[GTQuestion] = []
    for hebrew_domain, items in data.items():
        domain = DOMAIN_MAP.get(hebrew_domain, hebrew_domain)
        for item in items:
            questions.append(GTQuestion(
                question=item["שאלה"],
                expected_answer=item["תשובה"],
                expected_file=item["מקור"]["קובץ"],
                expected_page=item["מקור"]["עמוד"],
                domain=domain,
            ))

    logger.info(f"Loaded {len(questions)} ground truth questions from {path.name}")
    return questions


# ---------------------------------------------------------------------------
# Source file matching (4-tier fuzzy)
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    s = urllib.parse.unquote(s)
    s = s.lower().replace("-", "").replace(".", "").replace("_", "").replace(" ", "")
    return s


def match_source_file(citation_file: str, expected_file: str) -> bool:
    if not citation_file or not expected_file:
        return False

    # Tier 1: exact substring
    if expected_file in citation_file or citation_file in expected_file:
        return True

    # Tier 2: basename comparison
    cb = Path(citation_file).name
    eb = Path(expected_file).name
    if cb == eb:
        return True

    # Tier 3: normalized comparison
    if _normalize(cb) == _normalize(eb):
        return True

    # Tier 4: fuzzy ratio
    if SequenceMatcher(None, _normalize(cb), _normalize(eb)).ratio() >= 0.80:
        return True

    return False


# ---------------------------------------------------------------------------
# LLM answer scorer
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for an insurance chatbot. You compare the chatbot's answer against a ground truth answer and assign a correctness score.

Rules:
- If the ground truth is "לא" (no): the system must clearly indicate "no". If it says "yes" or confirms → score 0.0
- If the ground truth is "כן" (yes): the system must clearly confirm. If it denies → score 0.0
- If the ground truth contains a specific number or amount: the system must include that exact value
- If the ground truth is a longer explanation: the system must convey the same key facts (may add more detail)
- If the system says "I don't know" or equivalent (לא יודע, אין לי מידע) → score 0.0

Score scale:
- 1.0 = fully correct, covers all key points
- 0.7-0.9 = mostly correct, minor omissions
- 0.4-0.6 = partially correct, missing important details
- 0.1-0.3 = mostly wrong but has some relevant info
- 0.0 = completely wrong or "I don't know"

You MUST respond with valid JSON only: {"score": <float>, "reasoning": "<brief explanation>"}"""


def score_answer_correctness(
    question: str,
    expected: str,
    actual: str,
    llm: GeminiClient,
) -> tuple[float, str]:
    if not actual or not actual.strip():
        return 0.0, "Empty system answer"

    prompt = f"""Question: {question}

Ground truth answer: {expected}

System answer: {actual}

Compare the system answer against the ground truth and provide your score."""

    try:
        raw = llm.generate(prompt=prompt, system_prompt=JUDGE_SYSTEM_PROMPT, temperature=0.0, max_tokens=512)
        # Parse JSON from response
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = json.loads(raw)
        score = float(result.get("score", 0.0))
        reasoning = result.get("reasoning", "")
        return min(max(score, 0.0), 1.0), reasoning
    except Exception as e:
        logger.warning(f"LLM scoring failed: {e}")
        return 0.0, f"Scoring error: {e}"


# ---------------------------------------------------------------------------
# Single question evaluation
# ---------------------------------------------------------------------------

def evaluate_question(
    gt: GTQuestion,
    api: ChatbotAPIClient,
    llm: GeminiClient,
) -> GTQuestionResult:
    result = GTQuestionResult(
        question=gt.question,
        expected_answer=gt.expected_answer,
        expected_file=gt.expected_file,
        expected_page=gt.expected_page,
        domain=gt.domain,
    )

    # Query the API
    resp = api.ask(gt.question, language="he")
    result.system_answer = resp["answer"]
    result.system_citations = resp["citations"]
    result.system_domain = resp.get("domain")
    result.latency_s = resp["latency_s"]
    result.api_success = resp["success"]

    if not result.api_success:
        result.answer_reasoning = "API call failed"
        return result

    # Domain match
    if result.system_domain:
        result.domain_match = result.system_domain.lower() == gt.domain.lower()

    # File and page match — check all citations
    for citation in result.system_citations:
        cit_file = citation.get("file", citation.get("source", ""))
        cit_page = citation.get("page")

        if match_source_file(cit_file, gt.expected_file):
            result.file_match = True
            result.matched_citation_file = cit_file

            if cit_page is not None:
                try:
                    if int(cit_page) == int(gt.expected_page):
                        result.page_match = True
                        break  # Found exact match, no need to check more
                except (ValueError, TypeError):
                    pass

    # LLM answer scoring
    score, reasoning = score_answer_correctness(
        gt.question, gt.expected_answer, result.system_answer, llm,
    )
    result.answer_correctness = score
    result.answer_reasoning = reasoning

    return result


# ---------------------------------------------------------------------------
# Full evaluation run
# ---------------------------------------------------------------------------

def run_gt_evaluation(
    dataset_path: str | Path = "evaluation/dataset/ground_truth_test.json",
    api_base_url: str = "http://localhost:8000",
    api_timeout: float = 60.0,
    output_dir: str = "evaluation/reports",
) -> GTEvalResult:
    api = ChatbotAPIClient(base_url=api_base_url, timeout=api_timeout)
    llm = GeminiClient()

    # Health check
    if not api.health_check():
        logger.error("API health check failed. Is the server running?")
        api.close()
        return GTEvalResult()

    questions = load_ground_truth(dataset_path)
    if not questions:
        logger.error("No ground truth questions loaded.")
        api.close()
        return GTEvalResult()

    eval_result = GTEvalResult()
    start = time.time()

    for i, gt in enumerate(questions, 1):
        logger.info(f"[{i}/{len(questions)}] ({gt.domain}) {gt.question[:60]}...")
        qr = evaluate_question(gt, api, llm)
        eval_result.results.append(qr)

        status = "PASS" if qr.pass_all else "FAIL"
        logger.info(
            f"  → {status} | correctness={qr.answer_correctness:.2f} "
            f"file={qr.file_match} page={qr.page_match} domain={qr.domain_match}"
        )

    eval_result.total_time_s = time.time() - start
    api.close()

    # Save results
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = eval_result.to_dict()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    timestamped_path = out_dir / f"gt_eval_{timestamp}.json"
    latest_path = out_dir / "gt_eval_latest.json"

    for p in (timestamped_path, latest_path):
        with open(p, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"Report saved to {latest_path}")

    # Console summary
    s = report["summary"]
    print("\n" + "=" * 60)
    print("GROUND TRUTH EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Total questions:        {s['total_questions']}")
    print(f"  Passed:                 {s['passed']}")
    print(f"  Failed:                 {s['failed']}")
    print(f"  Pass rate:              {s['pass_rate']:.1%}")
    print(f"  Avg correctness:        {s['avg_answer_correctness']:.2f}")
    print(f"  File match rate:        {s['file_match_rate']:.1%}")
    print(f"  Page match rate:        {s['page_match_rate']:.1%}")
    print(f"  Domain match rate:      {s['domain_match_rate']:.1%}")
    print(f"  Avg latency:            {s['avg_latency_s']:.1f}s")
    print(f"  Total time:             {s['total_time_s']:.1f}s")

    print("\nPer-domain breakdown:")
    print(f"  {'Domain':<15} {'Count':>5} {'Pass%':>7} {'Correct':>8} {'File%':>7} {'Page%':>7}")
    print("  " + "-" * 50)
    for domain, ds in report["by_domain"].items():
        print(
            f"  {domain:<15} {ds['count']:>5} {ds['pass_rate']:>7.1%} "
            f"{ds['avg_answer_correctness']:>8.2f} {ds['file_match_rate']:>7.1%} "
            f"{ds['page_match_rate']:>7.1%}"
        )

    # List failed questions
    failed = [d for d in report["details"] if not d["pass_all"]]
    if failed:
        print(f"\nFailed questions ({len(failed)}):")
        for d in failed:
            reasons = []
            if d["answer_correctness"] < 0.5:
                reasons.append(f"correctness={d['answer_correctness']:.2f}")
            if not d["file_match"]:
                reasons.append("no file match")
            if not d["page_match"]:
                reasons.append("no page match")
            print(f"  - [{d['domain']}] {d['question'][:70]}...")
            print(f"    Reason: {', '.join(reasons)}")

    print("=" * 60 + "\n")

    return eval_result
