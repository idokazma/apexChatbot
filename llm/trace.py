"""Per-query LLM call trace collector using thread-local storage."""

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from loguru import logger

_local = threading.local()

TRACES_DIR = Path("data/traces")


@dataclass
class LLMCallRecord:
    """Record of a single LLM generate() call."""

    node: str
    prompt: str
    system_prompt: str
    response: str
    temperature: float
    max_tokens: int
    duration_ms: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class QueryTrace:
    """Full trace for a single query through the agent pipeline."""

    trace_id: str
    config: dict
    calls: list[LLMCallRecord] = field(default_factory=list)
    current_node: str = ""


def start_trace(config: dict) -> str:
    """Activate a new trace for the current thread. Returns trace_id."""
    trace_id = uuid.uuid4().hex[:12]
    _local.trace = QueryTrace(trace_id=trace_id, config=config)
    return trace_id


_progress_callback = None


def set_progress_callback(cb) -> None:
    """Set a callback function that fires when the current node changes.

    The callback receives (trace_id: str, node_name: str).
    """
    global _progress_callback
    _progress_callback = cb


def set_current_node(name: str) -> None:
    """Set the current node name for subsequent LLM calls."""
    trace = get_active_trace()
    if trace:
        trace.current_node = name
        if _progress_callback:
            _progress_callback(trace.trace_id, name)


def record_call(
    prompt: str,
    system_prompt: str,
    response: str,
    temperature: float,
    max_tokens: int,
    duration_ms: float,
) -> None:
    """Append an LLM call record to the active trace (if any)."""
    trace = get_active_trace()
    if trace is None:
        return

    trace.calls.append(
        LLMCallRecord(
            node=trace.current_node,
            prompt=prompt,
            system_prompt=system_prompt,
            response=response,
            temperature=temperature,
            max_tokens=max_tokens,
            duration_ms=round(duration_ms, 1),
        )
    )


def get_active_trace() -> QueryTrace | None:
    """Return the active trace for the current thread, or None."""
    return getattr(_local, "trace", None)


def end_trace() -> dict | None:
    """End the active trace and return it as a dict. Clears thread-local state."""
    trace = get_active_trace()
    if trace is None:
        return None

    result = {
        "trace_id": trace.trace_id,
        "config": trace.config,
        "llm_calls": [asdict(c) for c in trace.calls],
    }

    _local.trace = None
    return result


def save_trace(trace_data: dict) -> Path:
    """Save a completed trace dict to data/traces/{trace_id}.json."""
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    path = TRACES_DIR / f"{trace_data['trace_id']}.json"
    path.write_text(json.dumps(trace_data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.debug(
        "Trace saved: {path} ({n} LLM calls)",
        path=path,
        n=len(trace_data.get("llm_calls", [])),
    )
    return path
