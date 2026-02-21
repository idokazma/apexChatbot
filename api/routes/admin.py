"""Admin dashboard API endpoints: system status, document stats, live query logs."""

from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter
from loguru import logger
from sse_starlette.sse import EventSourceResponse

from api.dependencies import resources
from api.query_log import query_log
from pydantic import BaseModel

from api.schemas import (
    ComponentStatus,
    DocumentStatsResponse,
    DomainDocCount,
    QueryStatsResponse,
    SystemStatusResponse,
)
from config.domains import DOMAINS
from config.settings import settings

router = APIRouter(prefix="/admin", tags=["admin"])

_start_time = time.time()


# ── System Status ────────────────────────────────────────────────────────────


@router.get("/status", response_model=SystemStatusResponse)
async def system_status() -> SystemStatusResponse:
    """Return live status of every system component."""
    components: list[ComponentStatus] = []

    # 1. ChromaDB
    try:
        if resources.store:
            count = resources.store.get_count()
            components.append(
                ComponentStatus(
                    name="ChromaDB",
                    status="online",
                    detail=f"{count} chunks indexed",
                )
            )
        else:
            components.append(
                ComponentStatus(name="ChromaDB", status="offline", detail="Not initialized")
            )
    except Exception as exc:
        components.append(ComponentStatus(name="ChromaDB", status="offline", detail=str(exc)[:120]))

    # 2. Inference LLM (Ollama or Claude)
    try:
        if settings.inference_llm == "claude":
            components.append(
                ComponentStatus(
                    name="Inference LLM",
                    status="online" if resources.ollama_client else "offline",
                    detail="Claude API",
                )
            )
        elif resources.ollama_client and resources.ollama_client.is_available():
            components.append(
                ComponentStatus(
                    name="Inference LLM",
                    status="online",
                    detail=f"Ollama ({settings.ollama_model})",
                )
            )
        else:
            components.append(
                ComponentStatus(name="Inference LLM", status="offline", detail="Ollama not reachable")
            )
    except Exception as exc:
        components.append(ComponentStatus(name="Inference LLM", status="offline", detail=str(exc)[:120]))

    # 3. Embedding model
    try:
        if resources.embedding_model is not None:
            components.append(
                ComponentStatus(
                    name="Embedding Model",
                    status="online",
                    detail=settings.embedding_model,
                )
            )
        else:
            components.append(
                ComponentStatus(name="Embedding Model", status="offline", detail="Not loaded")
            )
    except Exception as exc:
        components.append(
            ComponentStatus(name="Embedding Model", status="offline", detail=str(exc)[:120])
        )

    # 4. Claude API (preprocessing)
    try:
        has_key = bool(settings.anthropic_api_key)
        components.append(
            ComponentStatus(
                name="Claude API",
                status="online" if has_key else "offline",
                detail="API key configured" if has_key else "No API key",
            )
        )
    except Exception as exc:
        components.append(
            ComponentStatus(name="Claude API", status="offline", detail=str(exc)[:120])
        )

    # 5. Reranker
    try:
        if resources.reranker is not None:
            components.append(
                ComponentStatus(name="Reranker", status="online", detail="Cross-encoder loaded")
            )
        else:
            components.append(
                ComponentStatus(name="Reranker", status="offline", detail="Not loaded")
            )
    except Exception as exc:
        components.append(ComponentStatus(name="Reranker", status="offline", detail=str(exc)[:120]))

    # 6. LangGraph Agent
    try:
        mode = settings.retrieval_mode
        if resources.agent is not None:
            components.append(
                ComponentStatus(
                    name="LangGraph Agent",
                    status="online",
                    detail=f"Mode: {mode}",
                )
            )
        else:
            components.append(
                ComponentStatus(name="LangGraph Agent", status="offline", detail="Not compiled")
            )
    except Exception as exc:
        components.append(
            ComponentStatus(name="LangGraph Agent", status="offline", detail=str(exc)[:120])
        )

    online = sum(1 for c in components if c.status == "online")
    total = len(components)
    if online == total:
        overall = "healthy"
    elif online == 0:
        overall = "down"
    else:
        overall = "degraded"

    offline_names = [c.name for c in components if c.status != "online"]
    if offline_names:
        logger.warning(
            "[ADMIN] Status check: {overall} | offline: {off}",
            overall=overall,
            off=", ".join(offline_names),
        )
    else:
        logger.debug("[ADMIN] Status check: healthy ({n}/{n} online)", n=total)

    return SystemStatusResponse(
        overall=overall,
        components=components,
        uptime_seconds=round(time.time() - _start_time, 1),
    )


# ── Inference LLM ────────────────────────────────────────────────────────────


class InferenceLLMRequest(BaseModel):
    llm: str  # "ollama" or "claude"


@router.get("/inference-llm")
async def get_inference_llm():
    """Return the current inference LLM."""
    return {"llm": settings.inference_llm}


@router.put("/inference-llm")
async def set_inference_llm(body: InferenceLLMRequest):
    """Hot-swap the inference LLM at runtime."""
    llm = body.llm
    if llm not in ("ollama", "claude"):
        return {"error": f"Unknown LLM: {llm}. Use 'ollama' or 'claude'."}

    if llm == settings.inference_llm:
        return {"llm": llm, "changed": False}

    resources.swap_inference_llm(llm)
    logger.info("[ADMIN] Inference LLM changed to '{llm}'", llm=llm)
    return {"llm": llm, "changed": True}


# ── Retrieval Mode ──────────────────────────────────────────────────────────


class RetrievalModeRequest(BaseModel):
    mode: str  # "rag", "agentic", or "combined"


@router.get("/retrieval-mode")
async def get_retrieval_mode():
    """Return the current retrieval mode."""
    return {"mode": settings.retrieval_mode}


@router.put("/retrieval-mode")
async def set_retrieval_mode(body: RetrievalModeRequest):
    """Hot-swap the retrieval mode at runtime."""
    mode = body.mode
    if mode not in ("rag", "agentic", "combined"):
        return {"error": f"Unknown mode: {mode}. Use 'rag', 'agentic', or 'combined'."}

    if mode == settings.retrieval_mode:
        return {"mode": mode, "changed": False}

    try:
        resources.swap_retrieval_mode(mode)
        logger.info("[ADMIN] Retrieval mode changed to '{mode}'", mode=mode)
        return {"mode": mode, "changed": True}
    except Exception as e:
        logger.error("[ADMIN] Failed to switch retrieval mode: {e}", e=str(e))
        return {"error": str(e)}


# ── Document Stats ───────────────────────────────────────────────────────────


@router.get("/documents", response_model=DocumentStatsResponse)
async def document_stats() -> DocumentStatsResponse:
    """Return per-domain document counts from ChromaDB."""
    domain_counts: list[DomainDocCount] = []
    total = 0

    if resources.store:
        try:
            total = resources.store.get_count()
        except Exception:
            total = 0

        for key, domain in DOMAINS.items():
            try:
                results = resources.store.collection.get(
                    where={"domain": key},
                    include=[],
                )
                count = len(results["ids"]) if results and results["ids"] else 0
            except Exception:
                count = 0
            domain_counts.append(DomainDocCount(domain=key, domain_he=domain.name_he, count=count))

    logger.debug(
        "[ADMIN] Document stats: {total} total chunks across {n} domains",
        total=total,
        n=len(domain_counts),
    )
    return DocumentStatsResponse(total_chunks=total, domains=domain_counts)


# ── Query Stats ──────────────────────────────────────────────────────────────


@router.get("/query-stats", response_model=QueryStatsResponse)
async def get_query_stats() -> QueryStatsResponse:
    """Return aggregate query statistics."""
    stats = query_log.get_stats()
    return QueryStatsResponse(**stats)


# ── Recent Query Logs ────────────────────────────────────────────────────────


@router.get("/logs")
async def recent_logs(limit: int = 50):
    """Return the most recent query log entries."""
    return query_log.get_recent(limit=limit)


@router.get("/logs/{log_id}")
async def log_detail(log_id: str):
    """Return the full response detail for a single query log entry."""
    entry = query_log.get_entry_detail(log_id)
    if entry is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Log entry not found")
    return entry


# ── Live Query Log SSE Stream ────────────────────────────────────────────────


@router.get("/logs/stream")
async def log_stream():
    """Server-Sent Events stream of live query logs."""

    async def event_generator():
        queue = query_log.subscribe()
        logger.info("[ADMIN] SSE client connected (live log stream)")
        try:
            while True:
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield {"event": "query", "data": data}
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield {"event": "ping", "data": ""}
        except asyncio.CancelledError:
            pass
        finally:
            query_log.unsubscribe(queue)
            logger.info("[ADMIN] SSE client disconnected")

    return EventSourceResponse(event_generator())
