"""Navigate node: wraps the agentic navigator as a LangGraph node.

Replaces the old route → retrieve → grade chain with a single node
that navigates the hierarchy top-down.
"""

from loguru import logger

from agent.state import AgentState
from retrieval.navigator.navigator import Navigator


def navigate_node(state: AgentState, navigator: Navigator) -> dict:
    """Navigate the hierarchy to find relevant chunks.

    Uses the rewritten query if available, else the original query.
    Returns retrieved_documents in the same format as the old retriever,
    so the generator node works without changes.
    """
    query = state.get("rewritten_query") or state["query"]
    language = state.get("detected_language", "he")

    logger.info(f"Navigating hierarchy for: '{query[:80]}...'")

    result = navigator.navigate(query, language)

    docs = result["retrieved_documents"]
    nav_path = result["navigation_path"]
    should_fallback = result["should_fallback"]

    trace = state.get("reasoning_trace", [])
    trace.extend(nav_path.get("trace", []))

    logger.info(
        f"Navigation complete: {len(docs)} chunks, "
        f"path: {len(nav_path.get('domains', []))} domains → "
        f"{len(nav_path.get('documents', []))} docs → "
        f"{len(nav_path.get('chunks', []))} chunks"
    )

    return {
        "retrieved_documents": docs,
        "graded_documents": docs,  # navigator pre-filters, so graded = retrieved
        "navigation_path": nav_path,
        "should_fallback": should_fallback,
        "reasoning_trace": trace,
    }
