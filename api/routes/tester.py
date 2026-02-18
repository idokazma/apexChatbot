"""Tester dashboard API: trigger quiz runs, track progress, browse reports."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

router = APIRouter(prefix="/tester", tags=["tester"])

# ── Directories ──────────────────────────────────────────────────────────────

QUIZ_REPORTS_DIR = Path("quizzer/reports")
EVAL_REPORTS_DIR = Path("evaluation/reports")


# ── Request / Response Models ────────────────────────────────────────────────


class QuizRunRequest(BaseModel):
    """Request body for triggering a quiz run."""

    num_questions: int = Field(default=50, ge=1, le=5000)
    docs_per_question: int = Field(default=2, ge=1, le=5)
    api_timeout: float = Field(default=60.0, ge=5, le=300)


class RunProgress(BaseModel):
    """Current quiz run progress."""

    running: bool = False
    phase: str = ""
    current: int = 0
    total: int = 0
    percent: float = 0.0
    avg_score: float = 0.0
    failures: int = 0
    elapsed_s: float = 0.0
    error: str | None = None
    completed: bool = False


class ReportSummary(BaseModel):
    """Metadata for a saved report."""

    filename: str
    type: str  # "quiz" | "eval" | "baseline"
    timestamp: float
    size_bytes: int


# ── In-Memory Run State ─────────────────────────────────────────────────────


@dataclass
class _RunState:
    """Mutable state for the background quiz run."""

    running: bool = False
    phase: str = ""
    current: int = 0
    total: int = 0
    avg_score: float = 0.0
    failures: int = 0
    start_time: float = 0.0
    error: str | None = None
    completed: bool = False
    # Keep last completed result data for quick access
    last_result_path: str | None = None
    subscribers: list = field(default_factory=list)

    def to_progress(self) -> RunProgress:
        elapsed = time.time() - self.start_time if self.running else 0.0
        pct = (self.current / self.total * 100) if self.total > 0 else 0.0
        return RunProgress(
            running=self.running,
            phase=self.phase,
            current=self.current,
            total=self.total,
            percent=round(pct, 1),
            avg_score=round(self.avg_score, 3),
            failures=self.failures,
            elapsed_s=round(elapsed, 1),
            error=self.error,
            completed=self.completed,
        )

    def notify(self) -> None:
        """Push progress to all SSE subscribers."""
        data = self.to_progress().model_dump_json()
        dead = []
        for q in self.subscribers:
            try:
                q.put_nowait(data)
            except Exception:
                dead.append(q)
        for q in dead:
            self.subscribers.remove(q)


_state = _RunState()
_lock = threading.Lock()


# ── Background Quiz Runner ───────────────────────────────────────────────────


def _run_quiz_background(num_questions: int, docs_per_question: int, api_timeout: float) -> None:
    """Execute a quiz run in a background thread, updating _state as we go."""
    from quizzer.runner import QuizRunConfig

    global _state

    config = QuizRunConfig(
        num_questions=num_questions,
        docs_per_question=docs_per_question,
        api_base_url="http://localhost:8000",
        api_timeout=api_timeout,
        output_dir=str(QUIZ_REPORTS_DIR),
        save_intermediate=True,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Connect to vector store
        _state.phase = "Connecting to vector store"
        _state.notify()

        from data_pipeline.store.vector_store import VectorStoreClient

        store = VectorStoreClient()
        store.connect()
        doc_count = store.get_count()
        logger.info(f"[TESTER] Vector store has {doc_count} documents")

        if doc_count == 0:
            _state.error = "No documents in vector store. Run the data pipeline first."
            _state.running = False
            _state.notify()
            return

        # Step 2: Sample documents
        _state.phase = "Sampling documents"
        _state.notify()

        from quizzer.document_sampler import sample_document_groups

        n_groups = int(num_questions * 1.5)
        doc_groups = sample_document_groups(
            store,
            n_groups=n_groups,
            docs_per_group=docs_per_question,
        )

        # Step 3: Generate questions
        _state.phase = "Generating questions"
        _state.notify()

        from llm.claude_client import ClaudeClient
        from quizzer.question_generator import generate_questions_batch

        claude = ClaudeClient()
        questions = generate_questions_batch(doc_groups, claude, target_count=num_questions)
        _state.total = len(questions)
        _state.notify()

        logger.info(f"[TESTER] Generated {len(questions)} questions")

        # Save questions
        q_data = [
            {
                "question": q.question,
                "question_type": q.question_type.value,
                "domain": q.domain,
                "language": q.language,
                "difficulty": q.difficulty,
                "expected_answer_hints": q.expected_answer_hints,
                "source_doc_titles": [d.get("source_doc_title", "") for d in q.source_documents],
            }
            for q in questions
        ]
        (output_dir / "generated_questions.json").write_text(
            json.dumps(q_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Step 4: Query & score
        _state.phase = "Testing questions"
        _state.notify()

        from quizzer.answer_scorer import QuizScore, score_answer
        from quizzer.api_client import ChatbotAPIClient

        api = ChatbotAPIClient(base_url=config.api_base_url, timeout=config.api_timeout)
        if not api.health_check():
            _state.error = "API is not healthy. Start the server with 'make serve' first."
            _state.running = False
            _state.notify()
            store.disconnect()
            return

        scores: list[QuizScore] = []
        api_failures = 0

        for i, question in enumerate(questions):
            api_response = api.ask(question.question, language=question.language)

            if not api_response.get("success"):
                api_failures += 1

            score = score_answer(question, api_response, claude)
            scores.append(score)

            _state.current = i + 1
            _state.failures = api_failures
            if scores:
                _state.avg_score = sum(s.overall_score for s in scores) / len(scores)
            _state.notify()

        api.close()

        # Step 5: Generate report
        _state.phase = "Generating report"
        _state.notify()

        # Build result object for report generation
        from quizzer.runner import QuizRunResult

        result = QuizRunResult(config=config)
        result.scores = scores
        result.questions_generated = len(questions)
        result.questions_asked = len(scores)
        result.api_failures = api_failures

        # Use a timestamped filename so we keep history
        ts = time.strftime("%Y%m%d_%H%M%S")

        # Save raw results
        raw_path = output_dir / f"quiz_results_{ts}.json"
        raw_path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # Also save as latest
        latest_path = output_dir / "quiz_results.json"
        latest_path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Generate HTML report
        from quizzer.report_generator import generate_report

        report_path = output_dir / f"quiz_report_{ts}.html"
        generate_report(result, report_path, claude)
        # Also save as latest
        latest_html = output_dir / "quiz_report.html"
        generate_report(result, latest_html, claude)

        store.disconnect()

        _state.phase = "Complete"
        _state.completed = True
        _state.running = False
        _state.last_result_path = str(raw_path)
        _state.notify()

        logger.info(f"[TESTER] Quiz run complete: {len(scores)} scores, avg={_state.avg_score:.3f}")

    except Exception as exc:
        logger.error(f"[TESTER] Quiz run failed: {exc}")
        _state.error = str(exc)
        _state.running = False
        _state.notify()


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.post("/run")
async def trigger_quiz_run(req: QuizRunRequest):
    """Trigger a new quiz run in the background."""
    with _lock:
        if _state.running:
            raise HTTPException(status_code=409, detail="A quiz run is already in progress.")
        _state.running = True
        _state.phase = "Starting"
        _state.current = 0
        _state.total = 0
        _state.avg_score = 0.0
        _state.failures = 0
        _state.start_time = time.time()
        _state.error = None
        _state.completed = False
        _state.last_result_path = None

    thread = threading.Thread(
        target=_run_quiz_background,
        args=(req.num_questions, req.docs_per_question, req.api_timeout),
        daemon=True,
    )
    thread.start()

    return {"status": "started", "num_questions": req.num_questions}


@router.get("/progress")
async def get_progress():
    """Return current quiz run progress."""
    return _state.to_progress()


@router.get("/progress/stream")
async def progress_stream():
    """SSE stream of quiz run progress updates."""

    async def event_generator():
        q: asyncio.Queue = asyncio.Queue(maxsize=50)
        _state.subscribers.append(q)
        logger.info("[TESTER] SSE client connected (progress stream)")
        try:
            # Send initial state
            yield {"event": "progress", "data": _state.to_progress().model_dump_json()}
            while True:
                try:
                    data = await asyncio.wait_for(q.get(), timeout=5.0)
                    yield {"event": "progress", "data": data}
                except asyncio.TimeoutError:
                    yield {"event": "ping", "data": ""}
        except asyncio.CancelledError:
            pass
        finally:
            if q in _state.subscribers:
                _state.subscribers.remove(q)
            logger.info("[TESTER] SSE client disconnected")

    return EventSourceResponse(event_generator())


@router.get("/reports")
async def list_reports():
    """List all saved quiz and evaluation reports."""
    reports: list[dict] = []

    for directory, report_type in [
        (QUIZ_REPORTS_DIR, "quiz"),
        (EVAL_REPORTS_DIR, "eval"),
    ]:
        if not directory.exists():
            continue
        for f in directory.iterdir():
            if f.suffix == ".json" and not f.name.startswith("intermediate"):
                reports.append(
                    {
                        "filename": f.name,
                        "path": str(f),
                        "type": report_type,
                        "timestamp": f.stat().st_mtime,
                        "size_bytes": f.stat().st_size,
                    }
                )

    reports.sort(key=lambda r: r["timestamp"], reverse=True)
    return reports


@router.get("/reports/{report_type}/{filename}")
async def get_report(report_type: str, filename: str):
    """Get the contents of a specific report file."""
    if report_type == "quiz":
        base_dir = QUIZ_REPORTS_DIR
    elif report_type == "eval":
        base_dir = EVAL_REPORTS_DIR
    else:
        raise HTTPException(status_code=400, detail="Invalid report type. Use 'quiz' or 'eval'.")

    # Prevent directory traversal
    if ".." in filename or "/" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    filepath = base_dir / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"Report not found: {filename}")

    try:
        data = json.loads(filepath.read_text(encoding="utf-8"))
        return data
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Report file is not valid JSON.")


@router.get("/eval-questions")
async def get_eval_questions():
    """Return the static evaluation question set."""
    path = Path("evaluation/dataset/questions.json")
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


@router.post("/run-eval")
async def trigger_eval_run():
    """Trigger the RAGAS evaluation harness in the background.

    This runs the fixed 12-question evaluation set through the agent directly.
    """
    with _lock:
        if _state.running:
            raise HTTPException(status_code=409, detail="A test run is already in progress.")
        _state.running = True
        _state.phase = "Starting evaluation"
        _state.current = 0
        _state.total = 0
        _state.avg_score = 0.0
        _state.failures = 0
        _state.start_time = time.time()
        _state.error = None
        _state.completed = False

    thread = threading.Thread(target=_run_eval_background, daemon=True)
    thread.start()

    return {"status": "started", "type": "eval"}


def _run_eval_background() -> None:
    """Run the RAGAS evaluation in a background thread."""
    try:
        _state.phase = "Loading resources"
        _state.notify()

        from agent.graph import create_agent_for_mode
        from config.settings import settings as app_settings
        from data_pipeline.embedder.embedding_model import EmbeddingModel
        from data_pipeline.store.vector_store import VectorStoreClient
        from evaluation.ragas_eval import run_evaluation
        from retrieval.reranker import Reranker

        mode = app_settings.retrieval_mode
        store = None
        embedding_model = None
        reranker = None

        if mode in ("rag", "combined"):
            store = VectorStoreClient()
            store.connect()
            embedding_model = EmbeddingModel()
            reranker = Reranker()

        agent = create_agent_for_mode(
            mode=mode,
            store=store,
            embedding_model=embedding_model,
            reranker=reranker,
            hierarchy_dir=app_settings.hierarchy_dir,
        )

        questions_path = Path("evaluation/dataset/questions.json")
        output_dir = Path("evaluation/reports")

        if not questions_path.exists():
            _state.error = "Questions file not found: evaluation/dataset/questions.json"
            _state.running = False
            _state.notify()
            return

        questions = json.loads(questions_path.read_text(encoding="utf-8"))
        _state.total = len(questions)
        _state.phase = "Running evaluation"
        _state.notify()

        scores = run_evaluation(agent, questions_path, output_dir)

        store.disconnect()

        _state.phase = "Complete"
        _state.completed = True
        _state.running = False
        _state.avg_score = scores.get("avg_weighted_score", 0.0)
        _state.current = _state.total
        _state.notify()

        logger.info(f"[TESTER] Eval complete: avg_weighted_score={_state.avg_score:.3f}")

    except Exception as exc:
        logger.error(f"[TESTER] Eval run failed: {exc}")
        _state.error = str(exc)
        _state.running = False
        _state.notify()
