"""Agent state definition for the LangGraph chatbot."""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State shared across all nodes in the agent graph."""

    # Conversation
    messages: Annotated[list, add_messages]
    query: str
    rewritten_query: str

    # Routing
    detected_domains: list[str]
    detected_language: str

    # Retrieval
    retrieved_documents: list[dict]
    graded_documents: list[dict]

    # Generation
    generation: str
    citations: list[dict]

    # Control flow
    is_grounded: bool
    retry_count: int
    should_fallback: bool
