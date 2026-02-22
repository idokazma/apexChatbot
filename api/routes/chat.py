"""Chat endpoint: main user-facing API."""

import asyncio
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from fastapi import APIRouter
from loguru import logger

from api.dependencies import resources
from api.query_log import QueryLogEntry, query_log
from api.schemas import ChatRequest, ChatResponse, Citation
from config.settings import settings
from llm.trace import end_trace, save_trace, start_trace

router = APIRouter()

# Thread pool for running agent queries without blocking the event loop
_agent_executor = ThreadPoolExecutor(max_workers=4)

# Simple in-memory conversation store (replace with Redis for production)
conversations: dict[str, list] = {}


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Process a chat message and return a grounded answer with citations."""
    conversation_id = request.conversation_id or uuid.uuid4().hex[:16]

    # Get or create conversation history
    history = conversations.get(conversation_id, [])

    # Build agent input
    agent_input = {
        "query": request.message,
        "messages": history,
        "rewritten_query": "",
        "detected_domains": [],
        "detected_language": request.language or "he",
        "navigation_path": {},
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

    # Run agent with per-query trace
    logger.info(f"Processing query: {request.message[:80]}...")
    t0 = time.time()
    error_msg = ""
    result = {}

    config_snapshot = {
        "retrieval_mode": settings.retrieval_mode,
        "inference_llm": settings.inference_llm,
        "ollama_model": settings.ollama_model,
        "gemini_model": settings.gemini_model,
        "embedding_model": settings.embedding_model,
        "top_k_retrieve": settings.top_k_retrieve,
        "top_k_rerank": settings.top_k_rerank,
    }

    def _run_agent_with_trace(agent_input: dict) -> tuple[dict, dict | None]:
        """Run agent in worker thread with trace collection."""
        trace_id = start_trace(config_snapshot)
        try:
            res = resources.agent.invoke(agent_input)
        except Exception:
            end_trace()
            raise
        trace_data = end_trace()
        return res, trace_data

    trace_data = None
    try:
        loop = asyncio.get_event_loop()
        result, trace_data = await loop.run_in_executor(
            _agent_executor, partial(_run_agent_with_trace, agent_input)
        )
    except Exception as exc:
        error_msg = str(exc)
        logger.error(f"Agent error: {exc}")

    duration_ms = round((time.time() - t0) * 1000, 1)

    # Extract response
    answer = result.get("generation", "")
    citations_raw = result.get("citations", [])
    domains = result.get("detected_domains", [])
    language = result.get("detected_language", "he")

    # Format citations
    citations = [
        Citation(
            source_url=c.get("source_url", ""),
            document_title=c.get("document_title", ""),
            section=c.get("section", ""),
            relevant_text=c.get("relevant_text", ""),
            page_number=c.get("page_number", 0),
            source_file_path=c.get("source_file_path", ""),
        )
        for c in citations_raw
    ]

    # Update conversation history
    from langchain_core.messages import AIMessage, HumanMessage

    history.append(HumanMessage(content=request.message))
    history.append(AIMessage(content=answer))
    conversations[conversation_id] = history[-10:]  # Keep last 5 turns

    # Calculate confidence
    is_grounded = result.get("is_grounded", False)
    is_fallback = result.get("should_fallback", False)
    confidence = 0.9 if is_grounded else (0.3 if is_fallback else 0.6)

    # Extract navigation path
    navigation_path = result.get("navigation_path", {})
    retrieval_mode = settings.retrieval_mode

    # Record query log for admin dashboard (includes full response for inspection)
    trace_id = trace_data.get("trace_id", "") if trace_data else ""
    query_log.record(
        QueryLogEntry(
            timestamp=t0,
            conversation_id=conversation_id,
            query=request.message,
            language=language,
            detected_domains=domains,
            rewritten_query=result.get("rewritten_query", ""),
            docs_retrieved=len(result.get("retrieved_documents", [])),
            docs_graded=len(result.get("graded_documents", [])),
            citations_count=len(citations),
            confidence=confidence,
            is_fallback=is_fallback,
            quality_action=result.get("quality_action", ""),
            duration_ms=duration_ms,
            error=error_msg,
            retrieval_mode=retrieval_mode,
            answer=answer,
            citations=[c.model_dump() for c in citations],
            trace_id=trace_id,
        )
    )

    # Save per-query trace JSON
    if trace_data:
        trace_data["timestamp"] = t0
        trace_data["query"] = request.message
        trace_data["result"] = {
            "answer": answer,
            "domains": domains,
            "confidence": confidence,
            "citations_count": len(citations),
            "duration_ms": duration_ms,
        }
        trace_data["reasoning_trace"] = result.get("reasoning_trace", [])
        try:
            save_trace(trace_data)
        except Exception as exc:
            logger.warning(f"Failed to save trace: {exc}")

    return ChatResponse(
        answer=answer,
        citations=citations,
        domain=domains[0] if domains else None,
        confidence=confidence,
        conversation_id=conversation_id,
        language=language,
        retrieval_mode=retrieval_mode,
        navigation_path=navigation_path,
    )
