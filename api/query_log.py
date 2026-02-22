"""Query log collector with SQLite persistence and SSE broadcast."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

from loguru import logger

# Database path — next to the project data directory
_DB_DIR = Path(__file__).resolve().parent.parent / "data"
_DB_PATH = _DB_DIR / "query_logs.db"


@dataclass
class QueryLogEntry:
    """A single query log record."""

    timestamp: float
    conversation_id: str
    query: str
    language: str
    detected_domains: list[str]
    rewritten_query: str
    docs_retrieved: int
    docs_graded: int
    citations_count: int
    confidence: float
    is_fallback: bool
    quality_action: str
    duration_ms: float
    error: str = ""
    retrieval_mode: str = "rag"
    # Full response fields for admin inspection
    answer: str = ""
    citations: list[dict] = field(default_factory=list)
    log_id: str = ""
    trace_id: str = ""

    def __post_init__(self) -> None:
        if not self.log_id:
            self.log_id = uuid.uuid4().hex[:12]

    def to_dict(self) -> dict:
        return asdict(self)

    def to_summary_dict(self) -> dict:
        """Return a lightweight dict without answer/citations for log tables."""
        d = asdict(self)
        d.pop("answer", None)
        d.pop("citations", None)
        # trace_id is kept so the UI can match completed rows to in-progress rows
        return d


def _init_db() -> sqlite3.Connection:
    """Create the SQLite database and table if they don't exist."""
    _DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS query_logs (
            log_id        TEXT PRIMARY KEY,
            timestamp     REAL NOT NULL,
            conversation_id TEXT,
            query         TEXT,
            language      TEXT,
            detected_domains TEXT,
            rewritten_query TEXT,
            docs_retrieved INTEGER,
            docs_graded   INTEGER,
            citations_count INTEGER,
            confidence    REAL,
            is_fallback   INTEGER,
            quality_action TEXT,
            duration_ms   REAL,
            error         TEXT,
            retrieval_mode TEXT,
            answer        TEXT,
            citations_json TEXT,
            trace_id      TEXT
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON query_logs (timestamp DESC)
    """)
    # Migration: add trace_id column if missing (existing databases)
    try:
        conn.execute("ALTER TABLE query_logs ADD COLUMN trace_id TEXT DEFAULT ''")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # column already exists
    conn.commit()
    return conn


@dataclass
class QueryLogCollector:
    """Ring-buffer of recent query logs with SQLite persistence and SSE broadcast."""

    max_entries: int = 500
    _entries: deque[QueryLogEntry] = field(default_factory=lambda: deque(maxlen=500))
    _subscribers: list[asyncio.Queue] = field(default_factory=list)
    _total_queries: int = 0
    _total_errors: int = 0
    _total_fallbacks: int = 0
    _total_duration_ms: float = 0.0
    _db: sqlite3.Connection | None = field(default=None, repr=False)
    _loop: asyncio.AbstractEventLoop | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        try:
            self._db = _init_db()
            # Load counters from DB
            row = self._db.execute(
                "SELECT COUNT(*), COALESCE(SUM(error != ''), 0), "
                "COALESCE(SUM(is_fallback), 0), COALESCE(SUM(duration_ms), 0) "
                "FROM query_logs"
            ).fetchone()
            if row:
                self._total_queries = row[0]
                self._total_errors = int(row[1])
                self._total_fallbacks = int(row[2])
                self._total_duration_ms = row[3]
            logger.info("[QUERY LOG] SQLite DB loaded: {n} historical entries", n=self._total_queries)
        except Exception as exc:
            logger.warning("[QUERY LOG] SQLite init failed, using in-memory only: {e}", e=exc)
            self._db = None

    def record(self, entry: QueryLogEntry) -> None:
        """Add a log entry, persist to DB, log to terminal, and notify SSE subscribers."""
        self._entries.append(entry)
        self._total_queries += 1
        self._total_duration_ms += entry.duration_ms
        if entry.error:
            self._total_errors += 1
        if entry.is_fallback:
            self._total_fallbacks += 1

        # Persist to SQLite
        self._persist(entry)

        # Terminal logging
        self._log_to_terminal(entry)

        # Notify all SSE subscribers (send summary without full answer for bandwidth)
        # Thread-safe: record() can be called from worker threads
        data = entry.to_summary_dict()
        data["log_id"] = entry.log_id
        dead: list[asyncio.Queue] = []
        for q in self._subscribers:
            if not self._enqueue_threadsafe(q, data):
                dead.append(q)
        for q in dead:
            self._subscribers.remove(q)

    def _persist(self, entry: QueryLogEntry) -> None:
        """Write an entry to SQLite."""
        if not self._db:
            return
        try:
            self._db.execute(
                "INSERT OR REPLACE INTO query_logs "
                "(log_id, timestamp, conversation_id, query, language, detected_domains, "
                "rewritten_query, docs_retrieved, docs_graded, citations_count, confidence, "
                "is_fallback, quality_action, duration_ms, error, retrieval_mode, answer, citations_json, trace_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    entry.log_id,
                    entry.timestamp,
                    entry.conversation_id,
                    entry.query,
                    entry.language,
                    json.dumps(entry.detected_domains, ensure_ascii=False),
                    entry.rewritten_query,
                    entry.docs_retrieved,
                    entry.docs_graded,
                    entry.citations_count,
                    entry.confidence,
                    1 if entry.is_fallback else 0,
                    entry.quality_action,
                    entry.duration_ms,
                    entry.error,
                    entry.retrieval_mode,
                    entry.answer,
                    json.dumps(entry.citations, ensure_ascii=False),
                    entry.trace_id,
                ),
            )
            self._db.commit()
        except Exception as exc:
            logger.warning("[QUERY LOG] SQLite write failed: {e}", e=exc)

    @staticmethod
    def _log_to_terminal(entry: QueryLogEntry) -> None:
        """Emit a structured log line to the terminal for each query."""
        domains_str = ", ".join(entry.detected_domains) if entry.detected_domains else "none"
        query_preview = entry.query[:80] + ("..." if len(entry.query) > 80 else "")

        if entry.error:
            logger.error(
                '[QUERY #{n}] ERROR | query="{q}" | error={err}',
                n=query_log._total_queries,
                q=query_preview,
                err=entry.error[:120],
            )
            return

        level = "warning" if entry.is_fallback else "info"
        status = "FALLBACK" if entry.is_fallback else "OK"

        getattr(logger, level)(
            "[QUERY #{n}] {status} | {dur}ms | "
            "domain={dom} | lang={lang} | "
            "retrieved={ret} graded={grad} citations={cit} | "
            "confidence={conf:.0%} | quality={qa} | "
            'query="{q}"',
            n=query_log._total_queries,
            status=status,
            dur=entry.duration_ms,
            dom=domains_str,
            lang=entry.language,
            ret=entry.docs_retrieved,
            grad=entry.docs_graded,
            cit=entry.citations_count,
            conf=entry.confidence,
            qa=entry.quality_action or "-",
            q=query_preview,
        )

    def _enqueue_threadsafe(self, q: asyncio.Queue, data: dict) -> bool:
        """Put data onto an asyncio.Queue from any thread (main or worker).

        Returns False if the queue is full (subscriber is too slow).
        """
        loop = self._loop
        if loop is None:
            # No event loop registered yet — drop the event
            return False
        try:
            loop.call_soon_threadsafe(q.put_nowait, data)
            return True
        except asyncio.QueueFull:
            return False
        except RuntimeError:
            # Loop closed
            return False

    def broadcast_progress(self, data: dict) -> None:
        """Send a progress event to all SSE subscribers without creating a log entry.

        Thread-safe: can be called from worker threads.
        """
        payload = {"_event": "progress", **data}
        dead: list[asyncio.Queue] = []
        for q in self._subscribers:
            if not self._enqueue_threadsafe(q, payload):
                dead.append(q)
        for q in dead:
            self._subscribers.remove(q)

    def subscribe(self) -> asyncio.Queue:
        """Create a new SSE subscriber queue and capture the running event loop."""
        # Capture the event loop so worker threads can enqueue thread-safely
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None
        q: asyncio.Queue = asyncio.Queue(maxsize=50)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        """Remove an SSE subscriber."""
        if q in self._subscribers:
            self._subscribers.remove(q)

    def get_recent(self, limit: int = 50) -> list[dict]:
        """Return the most recent N log entries as summary dicts."""
        # Prefer DB if available for persistence across restarts
        if self._db:
            try:
                rows = self._db.execute(
                    "SELECT log_id, timestamp, conversation_id, query, language, "
                    "detected_domains, rewritten_query, docs_retrieved, docs_graded, "
                    "citations_count, confidence, is_fallback, quality_action, "
                    "duration_ms, error, retrieval_mode, trace_id "
                    "FROM query_logs ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()
                results = []
                for r in reversed(rows):  # oldest first
                    results.append({
                        "log_id": r[0],
                        "timestamp": r[1],
                        "conversation_id": r[2],
                        "query": r[3],
                        "language": r[4],
                        "detected_domains": json.loads(r[5]) if r[5] else [],
                        "rewritten_query": r[6],
                        "docs_retrieved": r[7],
                        "docs_graded": r[8],
                        "citations_count": r[9],
                        "confidence": r[10],
                        "is_fallback": bool(r[11]),
                        "quality_action": r[12],
                        "duration_ms": r[13],
                        "error": r[14] or "",
                        "retrieval_mode": r[15] or "rag",
                        "trace_id": r[16] or "",
                    })
                return results
            except Exception as exc:
                logger.warning("[QUERY LOG] SQLite read failed: {e}", e=exc)
        # Fallback to in-memory
        entries = list(self._entries)[-limit:]
        return [e.to_summary_dict() for e in entries]

    def get_entry_detail(self, log_id: str) -> dict | None:
        """Return the full response detail for a single log entry by ID."""
        if self._db:
            try:
                row = self._db.execute(
                    "SELECT log_id, timestamp, conversation_id, query, language, "
                    "detected_domains, rewritten_query, docs_retrieved, docs_graded, "
                    "citations_count, confidence, is_fallback, quality_action, "
                    "duration_ms, error, retrieval_mode, answer, citations_json, trace_id "
                    "FROM query_logs WHERE log_id = ?",
                    (log_id,),
                ).fetchone()
                if row:
                    return {
                        "log_id": row[0],
                        "timestamp": row[1],
                        "conversation_id": row[2],
                        "query": row[3],
                        "language": row[4],
                        "detected_domains": json.loads(row[5]) if row[5] else [],
                        "rewritten_query": row[6],
                        "docs_retrieved": row[7],
                        "docs_graded": row[8],
                        "citations_count": row[9],
                        "confidence": row[10],
                        "is_fallback": bool(row[11]),
                        "quality_action": row[12],
                        "duration_ms": row[13],
                        "error": row[14] or "",
                        "retrieval_mode": row[15] or "rag",
                        "answer": row[16] or "",
                        "citations": json.loads(row[17]) if row[17] else [],
                        "trace_id": row[18] or "",
                    }
            except Exception as exc:
                logger.warning("[QUERY LOG] SQLite detail read failed: {e}", e=exc)
        # Fallback: search in-memory
        for entry in reversed(self._entries):
            if entry.log_id == log_id:
                return entry.to_dict()
        return None

    def get_stats(self) -> dict:
        """Aggregate statistics."""
        avg_duration = self._total_duration_ms / self._total_queries if self._total_queries else 0.0
        return {
            "total_queries": self._total_queries,
            "total_errors": self._total_errors,
            "total_fallbacks": self._total_fallbacks,
            "avg_duration_ms": round(avg_duration, 1),
        }


# Global singleton
query_log = QueryLogCollector()
