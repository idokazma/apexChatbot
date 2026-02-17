"""Chat endpoint: main user-facing API."""

import uuid

from fastapi import APIRouter
from loguru import logger

from api.dependencies import resources
from api.schemas import ChatRequest, ChatResponse, Citation

router = APIRouter()

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

    # Run agent
    logger.info(f"Processing query: {request.message[:80]}...")
    result = resources.agent.invoke(agent_input)

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

    return ChatResponse(
        answer=answer,
        citations=citations,
        domain=domains[0] if domains else None,
        confidence=confidence,
        conversation_id=conversation_id,
        language=language,
    )
