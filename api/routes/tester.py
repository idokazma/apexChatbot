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
class _QuestionState:
    """State for a single prepared question."""

    index: int
    question: str
    question_type: str
    domain: str
    language: str
    difficulty: str
    expected_answer_hints: str
    source_doc_titles: list[str]
    status: str = "pending"  # pending | running | completed | failed
    score: float | None = None
    latency_s: float | None = None
    result: dict | None = None  # Full score dict when completed

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "question": self.question,
            "question_type": self.question_type,
            "domain": self.domain,
            "language": self.language,
            "difficulty": self.difficulty,
            "expected_answer_hints": self.expected_answer_hints,
            "source_doc_titles": self.source_doc_titles,
            "status": self.status,
            "score": self.score,
            "latency_s": self.latency_s,
            "result": self.result,
        }


@dataclass
class _RunState:
    """Mutable state for the background quiz run."""

    running: bool = False
    phase: str = "idle"  # idle | preparing | ready | executing | done
    current: int = 0
    total: int = 0
    avg_score: float = 0.0
    failures: int = 0
    start_time: float = 0.0
    error: str | None = None
    completed: bool = False
    last_result_path: str | None = None
    # Two-phase state
    questions: list[_QuestionState] = field(default_factory=list)
    _generated_questions: list = field(default_factory=list)  # Raw GeneratedQuestion objects
    _config: object | None = None
    _is_ex2: bool = False
    _ex2_questions: list[str] = field(default_factory=list)
    _stop_requested: bool = False
    # SSE subscribers
    subscribers: list = field(default_factory=list)
    question_subscribers: list = field(default_factory=list)

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
        _push_to_queues(self.subscribers, data)

    def notify_question(self, question_state: _QuestionState) -> None:
        """Push per-question update to SSE subscribers."""
        data = json.dumps(question_state.to_dict())
        _push_to_queues(self.question_subscribers, data)

    def reset(self) -> None:
        """Reset state for a new run."""
        self.running = False
        self.phase = "idle"
        self.current = 0
        self.total = 0
        self.avg_score = 0.0
        self.failures = 0
        self.start_time = 0.0
        self.error = None
        self.completed = False
        self.last_result_path = None
        self.questions = []
        self._generated_questions = []
        self._config = None
        self._is_ex2 = False
        self._ex2_questions = []
        self._stop_requested = False


def _push_to_queues(queues: list, data: str) -> None:
    """Push data to async queues, removing dead ones."""
    dead = []
    for q in queues:
        try:
            q.put_nowait(data)
        except Exception:
            dead.append(q)
    for q in dead:
        queues.remove(q)


_state = _RunState()
_lock = threading.Lock()


# ── Background Workers ──────────────────────────────────────────────────────


def _prepare_background(num_questions: int, docs_per_question: int, api_timeout: float) -> None:
    """Phase 1: Generate questions in background thread."""
    from quizzer.runner import QuizRunConfig, prepare_questions

    config = QuizRunConfig(
        num_questions=num_questions,
        docs_per_question=docs_per_question,
        api_base_url="http://localhost:8000",
        api_timeout=api_timeout,
        output_dir=str(QUIZ_REPORTS_DIR),
        save_intermediate=True,
    )

    def progress_cb(phase: str, detail: str = "") -> None:
        _state.phase = phase if phase != "Ready" else "ready"
        _state.notify()

    try:
        questions = prepare_questions(config, progress_callback=progress_cb)

        # Auto-save quiz set for later reloading
        try:
            from quizzer.runner import save_quiz_set

            save_quiz_set(questions, QUIZ_REPORTS_DIR)
        except Exception as e:
            logger.warning(f"[TESTER] Failed to auto-save quiz set: {e}")

        # Store generated questions and build question state list
        _state._generated_questions = questions
        _state._config = config
        _state.total = len(questions)
        _state.questions = [
            _QuestionState(
                index=i,
                question=q.question,
                question_type=q.question_type.value,
                domain=q.domain,
                language=q.language,
                difficulty=q.difficulty,
                expected_answer_hints=q.expected_answer_hints,
                source_doc_titles=[d.get("source_doc_title", "") for d in q.source_documents],
            )
            for i, q in enumerate(questions)
        ]

        _state.phase = "ready"
        _state.running = False
        _state.notify()
        # Notify all question subscribers with the full list
        for qs in _state.questions:
            _state.notify_question(qs)

        logger.info(f"[TESTER] Preparation complete: {len(questions)} questions ready")

    except Exception as exc:
        logger.error(f"[TESTER] Preparation failed: {exc}")
        _state.error = str(exc)
        _state.running = False
        _state.phase = "idle"
        _state.notify()


def _execute_background() -> None:
    """Phase 2: Execute prepared questions in background thread."""
    from quizzer.runner import execute_questions

    questions = _state._generated_questions
    config = _state._config

    if not questions or not config:
        _state.error = "No prepared questions to execute."
        _state.running = False
        _state.phase = "ready"
        _state.notify()
        return

    def per_question_cb(index: int, score) -> None:
        if _state._stop_requested:
            raise InterruptedError("Stopped by user")

        qs = _state.questions[index]
        qs.status = "completed" if score.success else "failed"
        qs.score = round(score.overall_score, 3)
        qs.latency_s = round(score.latency_s, 1)
        qs.result = score.to_dict()
        _state.notify_question(qs)

        _state.current = index + 1
        if not score.success:
            _state.failures += 1
        # Update avg score
        completed_scores = [q.score for q in _state.questions if q.score is not None]
        if completed_scores:
            _state.avg_score = sum(completed_scores) / len(completed_scores)
        _state.notify()

        # Mark next question as running
        if index + 1 < len(_state.questions):
            next_qs = _state.questions[index + 1]
            next_qs.status = "running"
            _state.notify_question(next_qs)

    try:
        # Mark first question as running
        if _state.questions:
            _state.questions[0].status = "running"
            _state.notify_question(_state.questions[0])

        result = execute_questions(questions, config, per_question_callback=per_question_cb)

        _state.phase = "done"
        _state.completed = True
        _state.running = False
        _state.notify()

        logger.info(f"[TESTER] Execution complete: {len(result.scores)} scores, avg={_state.avg_score:.3f}")

    except InterruptedError:
        _state.phase = "done"
        _state.completed = True
        _state.running = False
        _state.error = "Stopped by user"
        _state.notify()
        logger.info(f"[TESTER] Execution stopped by user at {_state.current}/{_state.total}")

    except Exception as exc:
        logger.error(f"[TESTER] Execution failed: {exc}")
        _state.error = str(exc)
        _state.running = False
        _state.notify()


def _execute_ex2_background() -> None:
    """Execute ex2 evaluation questions (no LLM scoring, just API calls)."""
    from quizzer.api_client import ChatbotAPIClient

    questions = _state._ex2_questions
    if not questions:
        _state.error = "No ex2 questions loaded."
        _state.running = False
        _state.phase = "ready"
        _state.notify()
        return

    api = ChatbotAPIClient(base_url="http://localhost:8000", timeout=600.0)
    if not api.health_check():
        _state.error = "API is not healthy."
        _state.running = False
        _state.notify()
        return

    results = []

    try:
        # Mark first question as running
        if _state.questions:
            _state.questions[0].status = "running"
            _state.notify_question(_state.questions[0])

        for i, question_text in enumerate(questions):
            if _state._stop_requested:
                logger.info(f"[TESTER] Ex2 execution stopped by user at question {i}/{len(questions)}")
                break

            response = api.ask(question_text, language="he")

            qs = _state.questions[i]
            qs.latency_s = round(response.get("latency_s", 0), 1)
            qs.score = response.get("confidence", 0.0)

            if response.get("success"):
                qs.status = "completed"
                qs.result = {
                    "question": question_text,
                    "answer": response.get("answer", ""),
                    "domain": response.get("domain"),
                    "confidence": response.get("confidence", 0.0),
                    "citations": response.get("citations", []),
                    "latency_s": qs.latency_s,
                    "num_citations": len(response.get("citations", [])),
                    "overall_score": response.get("confidence", 0.0),
                    "success": True,
                }
            else:
                qs.status = "failed"
                qs.result = {
                    "question": question_text,
                    "answer": "(API failure)",
                    "success": False,
                    "latency_s": qs.latency_s,
                    "overall_score": 0.0,
                }
                _state.failures += 1

            _state.notify_question(qs)
            results.append(qs.result)

            _state.current = i + 1
            completed_scores = [q.score for q in _state.questions if q.score is not None]
            if completed_scores:
                _state.avg_score = sum(completed_scores) / len(completed_scores)
            _state.notify()

            # Mark next as running
            if i + 1 < len(_state.questions):
                _state.questions[i + 1].status = "running"
                _state.notify_question(_state.questions[i + 1])

        api.close()

        # Save results
        output_dir = Path("evaluation/reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        result_path = output_dir / f"ex2_results_{ts}.json"
        result_path.write_text(
            json.dumps({"results": results, "total": len(results)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        _state.phase = "done"
        _state.completed = True
        _state.running = False
        if _state._stop_requested:
            _state.error = "Stopped by user"
        _state.notify()

        logger.info(f"[TESTER] Ex2 execution {'stopped' if _state._stop_requested else 'complete'}: {len(results)} questions, avg_confidence={_state.avg_score:.3f}")

    except Exception as exc:
        logger.error(f"[TESTER] Ex2 execution failed: {exc}")
        _state.error = str(exc)
        _state.running = False
        _state.notify()


def _run_quiz_background(num_questions: int, docs_per_question: int, api_timeout: float) -> None:
    """Legacy: Execute a full quiz run (both phases) in a background thread."""
    from quizzer.runner import QuizRunConfig

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

        from quizzer.runner import QuizRunResult

        result = QuizRunResult(config=config)
        result.scores = scores
        result.questions_generated = len(questions)
        result.questions_asked = len(scores)
        result.api_failures = api_failures

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

        from quizzer.report_generator import generate_report

        report_path = output_dir / f"quiz_report_{ts}.html"
        generate_report(result, report_path, claude)
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


# ── Two-Phase Endpoints ─────────────────────────────────────────────────────


@router.post("/prepare")
async def prepare_questions(req: QuizRunRequest):
    """Phase 1: Generate questions only. Returns immediately, runs in background."""
    with _lock:
        if _state.running:
            raise HTTPException(status_code=409, detail="A run is already in progress.")
        _state.reset()
        _state.running = True
        _state.phase = "preparing"
        _state.start_time = time.time()

    thread = threading.Thread(
        target=_prepare_background,
        args=(req.num_questions, req.docs_per_question, req.api_timeout),
        daemon=True,
    )
    thread.start()

    return {"status": "preparing", "num_questions": req.num_questions}


@router.post("/execute")
async def execute_questions():
    """Phase 2: Execute prepared questions against the system."""
    with _lock:
        if _state.running:
            raise HTTPException(status_code=409, detail="A run is already in progress.")
        if _state.phase != "ready":
            raise HTTPException(
                status_code=400,
                detail=f"Cannot execute: phase is '{_state.phase}', expected 'ready'. Prepare questions first.",
            )
        if not _state._generated_questions and not _state._ex2_questions:
            raise HTTPException(status_code=400, detail="No prepared questions found.")
        _state.running = True
        _state.phase = "executing"
        _state.current = 0
        _state.avg_score = 0.0
        _state.failures = 0
        _state.start_time = time.time()
        _state.error = None
        _state.completed = False

    if _state._is_ex2:
        target = _execute_ex2_background
        num_q = len(_state._ex2_questions)
    else:
        target = _execute_background
        num_q = len(_state._generated_questions)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()

    return {"status": "executing", "num_questions": num_q}


@router.post("/stop")
async def stop_execution():
    """Request the running test to stop after the current question."""
    with _lock:
        if not _state.running:
            return {"stopped": False, "detail": "No run in progress."}
        _state._stop_requested = True
    logger.info("[TESTER] Stop requested by user")
    return {"stopped": True, "detail": "Stop requested. Will stop after current question."}


class RunSingleRequest(BaseModel):
    """Request to run a single question by index."""
    index: int = Field(ge=0)


@router.post("/run-single")
async def run_single_question(req: RunSingleRequest):
    """Execute a single question by index against the chatbot API."""
    with _lock:
        if _state.running:
            raise HTTPException(status_code=409, detail="A run is already in progress.")
        if _state.phase not in ("ready", "done"):
            raise HTTPException(
                status_code=400,
                detail=f"Cannot run single question: phase is '{_state.phase}'.",
            )
        if req.index >= len(_state.questions):
            raise HTTPException(status_code=400, detail=f"Index {req.index} out of range.")

    def _run_single_bg(idx: int) -> None:
        from quizzer.api_client import ChatbotAPIClient

        qs = _state.questions[idx]
        qs.status = "running"
        _state.notify_question(qs)

        api = ChatbotAPIClient(base_url="http://localhost:8000", timeout=600.0)

        if _state._is_ex2:
            question_text = _state._ex2_questions[idx]
            response = api.ask(question_text, language="he")
            qs.latency_s = round(response.get("latency_s", 0), 1)
            qs.score = response.get("confidence", 0.0)
            if response.get("success"):
                qs.status = "completed"
                qs.result = {
                    "question": question_text,
                    "answer": response.get("answer", ""),
                    "domain": response.get("domain"),
                    "confidence": response.get("confidence", 0.0),
                    "citations": response.get("citations", []),
                    "latency_s": qs.latency_s,
                    "num_citations": len(response.get("citations", [])),
                    "overall_score": response.get("confidence", 0.0),
                    "success": True,
                }
            else:
                qs.status = "failed"
                qs.result = {
                    "question": question_text,
                    "answer": "(API failure)",
                    "success": False,
                    "latency_s": qs.latency_s,
                    "overall_score": 0.0,
                }
        else:
            # Quiz question — use scorer
            from quizzer.answer_scorer import score_answer
            from llm.claude_client import ClaudeClient

            gq = _state._generated_questions[idx]
            api_response = api.ask(gq.question, language=gq.language)
            claude = ClaudeClient()
            score = score_answer(gq, api_response, claude)

            qs.latency_s = round(score.latency_s, 1)
            qs.score = round(score.overall_score, 3)
            qs.status = "completed" if score.success else "failed"
            qs.result = score.to_dict()

        api.close()
        _state.notify_question(qs)

        # Recalculate avg
        completed_scores = [q.score for q in _state.questions if q.score is not None]
        if completed_scores:
            _state.avg_score = sum(completed_scores) / len(completed_scores)
        _state.notify()
        logger.info(f"[TESTER] Single question {idx} completed: {qs.status}")

    thread = threading.Thread(target=_run_single_bg, args=(req.index,), daemon=True)
    thread.start()
    return {"status": "running", "index": req.index}


@router.get("/questions")
async def get_questions():
    """Return all prepared questions with their current status."""
    return {
        "phase": _state.phase,
        "total": len(_state.questions),
        "questions": [qs.to_dict() for qs in _state.questions],
    }


@router.get("/questions/stream")
async def questions_stream():
    """SSE stream of per-question status updates during execution."""

    async def event_generator():
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        _state.question_subscribers.append(q)
        logger.info("[TESTER] SSE client connected (questions stream)")
        try:
            # Send current state of all questions
            for qs in _state.questions:
                yield {"event": "question", "data": json.dumps(qs.to_dict())}
            while True:
                try:
                    data = await asyncio.wait_for(q.get(), timeout=5.0)
                    yield {"event": "question", "data": data}
                except asyncio.TimeoutError:
                    yield {"event": "ping", "data": ""}
        except asyncio.CancelledError:
            pass
        finally:
            if q in _state.question_subscribers:
                _state.question_subscribers.remove(q)
            logger.info("[TESTER] SSE client disconnected (questions stream)")

    return EventSourceResponse(event_generator())


# ── Quiz Set Endpoints ───────────────────────────────────────────────────────


class LoadQuizSetRequest(BaseModel):
    """Request body for loading a saved quiz set."""

    filename: str


@router.get("/quiz-sets")
async def list_quiz_sets():
    """List all saved quiz set files."""
    quiz_sets = []
    if not QUIZ_REPORTS_DIR.exists():
        return quiz_sets

    for f in QUIZ_REPORTS_DIR.iterdir():
        if f.name.startswith("quiz_set_") and f.suffix == ".json":
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                quiz_sets.append({
                    "filename": f.name,
                    "name": data.get("name", f.stem),
                    "created_at": data.get("created_at", ""),
                    "num_questions": data.get("num_questions", 0),
                    "size_bytes": f.stat().st_size,
                })
            except Exception:
                continue

    quiz_sets.sort(key=lambda x: x["created_at"], reverse=True)
    return quiz_sets


@router.post("/load-quiz-set")
async def load_quiz_set_endpoint(req: LoadQuizSetRequest):
    """Load a saved quiz set into state, ready for execution."""
    with _lock:
        if _state.running:
            raise HTTPException(status_code=409, detail="A run is already in progress.")

    if ".." in req.filename or "/" in req.filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    filepath = QUIZ_REPORTS_DIR / req.filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"Quiz set not found: {req.filename}")

    try:
        from quizzer.runner import QuizRunConfig, load_quiz_set

        questions = load_quiz_set(filepath)

        if not questions:
            raise HTTPException(status_code=400, detail="Quiz set contains no questions.")

        config = QuizRunConfig(
            num_questions=len(questions),
            api_base_url="http://localhost:8000",
            output_dir=str(QUIZ_REPORTS_DIR),
            save_intermediate=True,
        )

        with _lock:
            _state.reset()
            _state._generated_questions = questions
            _state._config = config
            _state.total = len(questions)
            _state.phase = "ready"
            _state.questions = [
                _QuestionState(
                    index=i,
                    question=q.question,
                    question_type=q.question_type.value,
                    domain=q.domain,
                    language=q.language,
                    difficulty=q.difficulty,
                    expected_answer_hints=q.expected_answer_hints,
                    source_doc_titles=[d.get("source_doc_title", "") for d in q.source_documents],
                )
                for i, q in enumerate(questions)
            ]

        logger.info(f"[TESTER] Loaded quiz set '{req.filename}' with {len(questions)} questions")
        return {
            "status": "ready",
            "num_questions": len(questions),
            "filename": req.filename,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[TESTER] Failed to load quiz set: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load quiz set: {e}")


# ── Ex2 Evaluation Endpoints ──────────────────────────────────────────────────


EX2_QUESTIONS_PATH = Path("ex2_evaluation_script/questions.txt")


class LoadEx2Request(BaseModel):
    """Optional limit for ex2 questions."""

    limit: int | None = Field(default=None, ge=1)


@router.post("/load-ex2")
async def load_ex2_questions(req: LoadEx2Request | None = None):
    """Load the ex2 evaluation questions (120 Hebrew questions, no ground truth)."""
    with _lock:
        if _state.running:
            raise HTTPException(status_code=409, detail="A run is already in progress.")

    if not EX2_QUESTIONS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Ex2 questions file not found: {EX2_QUESTIONS_PATH}",
        )

    try:
        questions = json.loads(EX2_QUESTIONS_PATH.read_text(encoding="utf-8"))
        if not isinstance(questions, list) or not questions:
            raise HTTPException(status_code=400, detail="Ex2 questions file is empty or invalid.")

        limit = req.limit if req and req.limit else None
        if limit:
            questions = questions[:limit]

        with _lock:
            _state.reset()
            _state._is_ex2 = True
            _state._ex2_questions = questions
            _state.total = len(questions)
            _state.phase = "ready"
            _state.questions = [
                _QuestionState(
                    index=i,
                    question=q,
                    question_type="ex2",
                    domain="-",
                    language="he",
                    difficulty="-",
                    expected_answer_hints="",
                    source_doc_titles=[],
                )
                for i, q in enumerate(questions)
            ]

        logger.info(f"[TESTER] Loaded {len(questions)} ex2 evaluation questions")
        return {"status": "ready", "num_questions": len(questions)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[TESTER] Failed to load ex2 questions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load ex2 questions: {e}")


# ── Legacy Endpoints (kept for backward compat) ─────────────────────────────


@router.post("/run")
async def trigger_quiz_run(req: QuizRunRequest):
    """Trigger a new quiz run in the background (legacy single-phase)."""
    with _lock:
        if _state.running:
            raise HTTPException(status_code=409, detail="A quiz run is already in progress.")
        _state.reset()
        _state.running = True
        _state.phase = "Starting"
        _state.start_time = time.time()

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
    """Trigger the RAGAS evaluation harness in the background."""
    with _lock:
        if _state.running:
            raise HTTPException(status_code=409, detail="A test run is already in progress.")
        _state.reset()
        _state.running = True
        _state.phase = "Starting evaluation"
        _state.start_time = time.time()

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
