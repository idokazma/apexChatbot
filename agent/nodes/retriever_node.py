"""Retrieval node: fetches relevant documents from the vector store."""

from loguru import logger

from agent.state import AgentState
from retrieval.retriever import Retriever


def retriever_node(state: AgentState, retriever: Retriever) -> dict:
    """Retrieve relevant documents based on the query and detected domains."""
    query = state.get("rewritten_query") or state["query"]
    domains = state.get("detected_domains", [])

    results = retriever.retrieve(
        query=query,
        domains=domains if domains else None,
    )

    logger.info(f"Retrieved {len(results)} documents")
    return {"retrieved_documents": results}
