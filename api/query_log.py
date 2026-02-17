"""In-memory query log collector for the admin dashboard."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import asdict, dataclass, field

from loguru import logger


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

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QueryLogCollector:
    """Thread-safe ring-buffer of recent query logs with SSE broadcast."""

    max_entries: int = 500
    _entries: deque[QueryLogEntry] = field(default_factory=lambda: deque(maxlen=500))
    _subscribers: list[asyncio.Queue] = field(default_factory=list)
    _total_queries: int = 0
    _total_errors: int = 0
    _total_fallbacks: int = 0
    _total_duration_ms: float = 0.0

    def record(self, entry: QueryLogEntry) -> None:
        """Add a log entry, log to terminal, and notify SSE subscribers."""
        self._entries.append(entry)
        self._total_queries += 1
        self._total_duration_ms += entry.duration_ms
        if entry.error:
            self._total_errors += 1
        if entry.is_fallback:
            self._total_fallbacks += 1

        # Terminal logging
        self._log_to_terminal(entry)

        # Notify all SSE subscribers
        data = entry.to_dict()
        dead: list[asyncio.Queue] = []
        for q in self._subscribers:
            try:
                q.put_nowait(data)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            self._subscribers.remove(q)

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

    def subscribe(self) -> asyncio.Queue:
        """Create a new SSE subscriber queue."""
        q: asyncio.Queue = asyncio.Queue(maxsize=50)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        """Remove an SSE subscriber."""
        if q in self._subscribers:
            self._subscribers.remove(q)

    def get_recent(self, limit: int = 50) -> list[dict]:
        """Return the most recent N log entries as dicts."""
        entries = list(self._entries)[-limit:]
        return [e.to_dict() for e in entries]

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
