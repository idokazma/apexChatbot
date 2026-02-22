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

# ── Prompt Registry ─────────────────────────────────────────────────────────
# Maps prompt_id → (module_path, variable_name, display_name, description)

_PROMPT_REGISTRY: list[dict] = [
    {
        "id": "system_prompt",
        "name": "System Prompt (English)",
        "description": "Main system instruction for English queries",
        "module": "config.prompts.system_prompt",
        "variable": "SYSTEM_PROMPT",
        "stage": "generator",
    },
    {
        "id": "system_prompt_he",
        "name": "System Prompt (Hebrew)",
        "description": "Main system instruction for Hebrew queries",
        "module": "config.prompts.system_prompt",
        "variable": "SYSTEM_PROMPT_HE",
        "stage": "generator",
    },
    {
        "id": "routing_prompt",
        "name": "Domain Routing",
        "description": "Classifies query into insurance domain(s)",
        "module": "config.prompts.routing_prompt",
        "variable": "ROUTING_PROMPT",
        "stage": "router",
    },
    {
        "id": "query_rewrite_prompt",
        "name": "Query Rewrite",
        "description": "Rewrites user query for better retrieval",
        "module": "config.prompts.routing_prompt",
        "variable": "QUERY_REWRITE_PROMPT",
        "stage": "query_analyzer",
    },
    {
        "id": "relevance_grading_prompt",
        "name": "Relevance Grading",
        "description": "Assesses document relevance to query",
        "module": "config.prompts.grading_prompt",
        "variable": "RELEVANCE_GRADING_PROMPT",
        "stage": "grader",
    },
    {
        "id": "generation_prompt",
        "name": "Answer Generation",
        "description": "Generates final answer from retrieved docs",
        "module": "config.prompts.grading_prompt",
        "variable": "GENERATION_PROMPT",
        "stage": "generator",
    },
    {
        "id": "hallucination_check_prompt",
        "name": "Hallucination Check",
        "description": "Verifies answer is grounded in sources",
        "module": "config.prompts.grading_prompt",
        "variable": "HALLUCINATION_CHECK_PROMPT",
        "stage": "quality",
    },
    {
        "id": "quality_check_prompt",
        "name": "Quality Check",
        "description": "Verifies answer quality, can reroute/rephrase",
        "module": "agent.nodes.quality_checker",
        "variable": "_QUALITY_CHECK_PROMPT",
        "stage": "quality",
    },
    {
        "id": "domain_selection_prompt",
        "name": "Domain Selection (Navigator)",
        "description": "Selects domain(s) via hierarchical navigation",
        "module": "retrieval.navigator.navigator_prompts",
        "variable": "DOMAIN_SELECTION_PROMPT",
        "stage": "navigator",
    },
    {
        "id": "document_selection_prompt",
        "name": "Document Selection (Navigator)",
        "description": "Selects documents within a domain",
        "module": "retrieval.navigator.navigator_prompts",
        "variable": "DOCUMENT_SELECTION_PROMPT",
        "stage": "navigator",
    },
]


def _get_prompt_value(module_path: str, variable: str) -> str:
    """Import and return current value of a prompt variable."""
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, variable, "")


def _set_prompt_value(module_path: str, variable: str, value: str) -> None:
    """Set a prompt variable at runtime (in-memory only)."""
    import importlib
    mod = importlib.import_module(module_path)
    setattr(mod, variable, value)

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

    # 2. Inference LLM (Ollama, Claude, or Gemini)
    try:
        if settings.inference_llm == "claude":
            components.append(
                ComponentStatus(
                    name="Inference LLM",
                    status="online" if resources.ollama_client else "offline",
                    detail="Claude API",
                )
            )
        elif settings.inference_llm == "gemini":
            components.append(
                ComponentStatus(
                    name="Inference LLM",
                    status="online" if resources.ollama_client else "offline",
                    detail=f"Gemini ({settings.gemini_model})",
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

    # 4. Preprocessing APIs (Claude + Gemini keys)
    try:
        has_claude = bool(settings.anthropic_api_key)
        has_gemini = bool(settings.google_api_key)
        keys = []
        if has_claude:
            keys.append("Claude")
        if has_gemini:
            keys.append("Gemini")
        components.append(
            ComponentStatus(
                name="Preprocessing APIs",
                status="online" if keys else "offline",
                detail=", ".join(keys) + " configured" if keys else "No API keys",
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
    llm: str  # "ollama", "claude", or "gemini"


@router.get("/inference-llm")
async def get_inference_llm():
    """Return the current inference LLM."""
    return {"llm": settings.inference_llm}


@router.put("/inference-llm")
async def set_inference_llm(body: InferenceLLMRequest):
    """Hot-swap the inference LLM at runtime."""
    llm = body.llm
    if llm not in ("ollama", "claude", "gemini"):
        return {"error": f"Unknown LLM: {llm}. Use 'ollama', 'claude', or 'gemini'."}

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


# ── Live Query Log SSE Stream ────────────────────────────────────────────────
# NOTE: /logs/stream MUST be defined before /logs/{log_id} to avoid
# FastAPI matching "stream" as a log_id parameter.


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


@router.get("/logs/{log_id}")
async def log_detail(log_id: str):
    """Return the full response detail for a single query log entry."""
    entry = query_log.get_entry_detail(log_id)
    if entry is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Log entry not found")
    return entry


# ── Prompts ─────────────────────────────────────────────────────────────────


@router.get("/prompts")
async def list_prompts():
    """Return all prompt templates with their current values."""
    result = []
    for entry in _PROMPT_REGISTRY:
        result.append({
            "id": entry["id"],
            "name": entry["name"],
            "description": entry["description"],
            "stage": entry["stage"],
            "content": _get_prompt_value(entry["module"], entry["variable"]),
        })
    return result


class PromptUpdateRequest(BaseModel):
    content: str


# ── Agent Graph Topology ────────────────────────────────────────────────────


_GRAPH_TOPOLOGIES = {
    "rag": {
        "nodes": [
            {"id": "analyze", "label": "Query Analyzer", "type": "process"},
            {"id": "route", "label": "Router", "type": "decision"},
            {"id": "retrieve", "label": "Retriever", "type": "process"},
            {"id": "grade", "label": "Grader", "type": "decision"},
            {"id": "generate", "label": "Generator", "type": "process"},
            {"id": "quality_check", "label": "Quality Check", "type": "decision"},
            {"id": "increment_retry", "label": "Retry", "type": "control"},
            {"id": "fallback", "label": "Fallback", "type": "terminal"},
            {"id": "end", "label": "Done", "type": "terminal"},
        ],
        "edges": [
            {"from": "analyze", "to": "route"},
            {"from": "route", "to": "retrieve", "label": "on-topic"},
            {"from": "route", "to": "fallback", "label": "off-topic"},
            {"from": "retrieve", "to": "grade"},
            {"from": "grade", "to": "generate", "label": "relevant"},
            {"from": "grade", "to": "increment_retry", "label": "retry"},
            {"from": "grade", "to": "fallback", "label": "no docs"},
            {"from": "generate", "to": "quality_check"},
            {"from": "quality_check", "to": "end", "label": "pass"},
            {"from": "quality_check", "to": "increment_retry", "label": "reroute/rephrase"},
            {"from": "quality_check", "to": "fallback", "label": "fail"},
            {"from": "increment_retry", "to": "retrieve"},
            {"from": "fallback", "to": "end"},
        ],
    },
    "agentic": {
        "nodes": [
            {"id": "analyze", "label": "Query Analyzer", "type": "process"},
            {"id": "navigate", "label": "Navigator", "type": "process"},
            {"id": "generate", "label": "Generator", "type": "process"},
            {"id": "quality_check", "label": "Quality Check", "type": "decision"},
            {"id": "increment_retry", "label": "Retry", "type": "control"},
            {"id": "fallback", "label": "Fallback", "type": "terminal"},
            {"id": "end", "label": "Done", "type": "terminal"},
        ],
        "edges": [
            {"from": "analyze", "to": "navigate"},
            {"from": "navigate", "to": "generate", "label": "docs found"},
            {"from": "navigate", "to": "fallback", "label": "no docs"},
            {"from": "generate", "to": "quality_check"},
            {"from": "quality_check", "to": "end", "label": "pass"},
            {"from": "quality_check", "to": "increment_retry", "label": "reroute/rephrase"},
            {"from": "quality_check", "to": "fallback", "label": "fail"},
            {"from": "increment_retry", "to": "navigate"},
            {"from": "fallback", "to": "end"},
        ],
    },
    "combined": {
        "nodes": [
            {"id": "analyze", "label": "Query Analyzer", "type": "process"},
            {"id": "combined_retrieve", "label": "Combined Retriever", "type": "process"},
            {"id": "generate", "label": "Generator", "type": "process"},
            {"id": "quality_check", "label": "Quality Check", "type": "decision"},
            {"id": "increment_retry", "label": "Retry", "type": "control"},
            {"id": "fallback", "label": "Fallback", "type": "terminal"},
            {"id": "end", "label": "Done", "type": "terminal"},
        ],
        "edges": [
            {"from": "analyze", "to": "combined_retrieve"},
            {"from": "combined_retrieve", "to": "generate", "label": "docs found"},
            {"from": "combined_retrieve", "to": "fallback", "label": "no docs"},
            {"from": "generate", "to": "quality_check"},
            {"from": "quality_check", "to": "end", "label": "pass"},
            {"from": "quality_check", "to": "increment_retry", "label": "reroute/rephrase"},
            {"from": "quality_check", "to": "fallback", "label": "fail"},
            {"from": "increment_retry", "to": "combined_retrieve"},
            {"from": "fallback", "to": "end"},
        ],
    },
}


_NODE_DETAILS = {
    "analyze": {
        "label": "Query Analyzer",
        "steps": [
            "Detect query language (Hebrew/English) via langdetect",
            "Extract last 2 conversation turns for context",
            "Rewrite query for better retrieval using LLM",
        ],
        "prompts": ["query_rewrite_prompt"],
        "reads": ["query", "messages"],
        "writes": ["detected_language", "rewritten_query"],
    },
    "route": {
        "label": "Router",
        "steps": [
            "Phase 1: Fast keyword matching with regex patterns per domain",
            "If keywords match → return detected domains immediately",
            "Phase 2 (fallback): LLM classifies ambiguous queries into domains",
            "Check for off_topic indicator → trigger fallback if so",
        ],
        "prompts": ["routing_prompt"],
        "reads": ["query", "rewritten_query"],
        "writes": ["detected_domains", "should_fallback"],
    },
    "retrieve": {
        "label": "Retriever",
        "steps": [
            "Embed rewritten query using multilingual-e5-large",
            "Vector search in ChromaDB with optional domain filter",
            "Rerank results with cross-encoder model",
            "Return top-k documents",
        ],
        "prompts": [],
        "reads": ["rewritten_query", "detected_domains"],
        "writes": ["retrieved_documents"],
    },
    "grade": {
        "label": "Grader",
        "steps": [
            "For each retrieved document:",
            "  Truncate content to 2000 chars",
            "  LLM grades relevance (yes/no) against the query",
            "Filter to keep only relevant documents",
        ],
        "prompts": ["relevance_grading_prompt"],
        "reads": ["query", "retrieved_documents"],
        "writes": ["graded_documents"],
    },
    "generate": {
        "label": "Generator",
        "steps": [
            "Format graded documents as numbered context",
            "Select system prompt based on language (HE/EN)",
            "Generate answer with LLM (temp=0.1)",
            "Extract citations: [1], [2] references + [Source: ...] patterns",
            "Deduplicate citations by URL/title",
        ],
        "prompts": ["system_prompt", "system_prompt_he", "generation_prompt"],
        "reads": ["query", "graded_documents", "detected_language"],
        "writes": ["generation", "citations"],
    },
    "quality_check": {
        "label": "Quality Check",
        "steps": [
            "Format source document summaries",
            "LLM evaluates answer quality against sources",
            "Parse action: PASS / REROUTE / REPHRASE / FAIL",
            "PASS → end, REROUTE → retry with new domain, REPHRASE → retry with new query",
        ],
        "prompts": ["quality_check_prompt"],
        "reads": ["query", "generation", "graded_documents", "detected_domains"],
        "writes": ["is_grounded", "quality_action", "quality_reasoning"],
    },
    "navigate": {
        "label": "Navigator",
        "steps": [
            "Level 0: LLM selects domain(s) from catalog summaries",
            "Level 1: LLM selects documents from domain shelf",
            "Level 2: Retrieve chunks from selected document cards",
            "Merge results with navigation breadcrumb trace",
        ],
        "prompts": ["domain_selection_prompt", "document_selection_prompt"],
        "reads": ["rewritten_query", "detected_language"],
        "writes": ["retrieved_documents", "graded_documents", "navigation_path", "should_fallback"],
    },
    "combined_retrieve": {
        "label": "Combined Retriever",
        "steps": [
            "Navigator path: traverse hierarchy (catalog → shelf → cards)",
            "RAG path: vector search + rerank in parallel",
            "Merge results: navigator docs first, then RAG docs",
            "Deduplicate by chunk_id, tag source (_source: navigator/rag)",
        ],
        "prompts": ["domain_selection_prompt", "document_selection_prompt"],
        "reads": ["rewritten_query", "detected_domains", "detected_language"],
        "writes": ["retrieved_documents", "graded_documents", "navigation_path", "should_fallback"],
    },
    "increment_retry": {
        "label": "Retry",
        "steps": [
            "Increment retry counter",
            "Preserve rerouted domains or rephrased query from quality check",
            "Append retry reason to reasoning trace",
        ],
        "prompts": [],
        "reads": ["retry_count", "quality_action", "quality_reasoning"],
        "writes": ["retry_count", "reasoning_trace"],
    },
    "fallback": {
        "label": "Fallback",
        "steps": [
            "Select fallback message based on language (HE/EN)",
            "If partial documents exist, append summaries of top 2",
            "Return safe no-evidence response with empty citations",
        ],
        "prompts": [],
        "reads": ["detected_language", "graded_documents"],
        "writes": ["generation", "citations", "should_fallback"],
    },
    "end": {
        "label": "Done",
        "steps": ["Return final answer to the user"],
        "prompts": [],
        "reads": [],
        "writes": [],
    },
}


@router.get("/graph-topology")
async def graph_topology():
    """Return the node/edge topology for the current agent graph mode."""
    mode = settings.retrieval_mode
    topo = _GRAPH_TOPOLOGIES.get(mode, _GRAPH_TOPOLOGIES["rag"])
    # Attach detail info to each node
    enriched_nodes = []
    for node in topo["nodes"]:
        detail = _NODE_DETAILS.get(node["id"], {})
        enriched_nodes.append({**node, **detail})
    return {"mode": mode, "nodes": enriched_nodes, "edges": topo["edges"]}


# ── Traces ──────────────────────────────────────────────────────────────────


@router.get("/traces")
async def list_traces(limit: int = 50):
    """Return the most recent query trace files (newest first)."""
    import json
    from llm.trace import TRACES_DIR

    if not TRACES_DIR.exists():
        return []

    files = sorted(TRACES_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    results = []
    for f in files[:limit]:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            results.append({
                "trace_id": data.get("trace_id", f.stem),
                "timestamp": data.get("timestamp"),
                "query": data.get("query", ""),
                "llm_calls_count": len(data.get("llm_calls", [])),
                "duration_ms": data.get("result", {}).get("duration_ms"),
                "config": data.get("config", {}),
            })
        except Exception:
            continue
    return results


@router.get("/traces/{trace_id}")
async def trace_detail(trace_id: str):
    """Return the full trace JSON for a single query."""
    import json
    from llm.trace import TRACES_DIR

    path = TRACES_DIR / f"{trace_id}.json"
    if not path.exists():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Trace not found")

    return json.loads(path.read_text(encoding="utf-8"))


@router.put("/prompts/{prompt_id}")
async def update_prompt(prompt_id: str, body: PromptUpdateRequest):
    """Update a prompt template at runtime (in-memory)."""
    entry = next((e for e in _PROMPT_REGISTRY if e["id"] == prompt_id), None)
    if not entry:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Prompt not found: {prompt_id}")

    _set_prompt_value(entry["module"], entry["variable"], body.content)
    logger.info(f"[ADMIN] Prompt '{prompt_id}' updated ({len(body.content)} chars)")
    return {"id": prompt_id, "updated": True}
