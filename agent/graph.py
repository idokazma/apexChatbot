"""LangGraph agent definition: the core chatbot orchestration graph."""

from functools import partial

from langgraph.graph import END, StateGraph
from loguru import logger
from pymilvus import Collection

from agent.nodes.fallback import fallback
from agent.nodes.generator import generator
from agent.nodes.grader import grader
from agent.nodes.hallucination_checker import hallucination_checker
from agent.nodes.query_analyzer import query_analyzer
from agent.nodes.retriever_node import retriever_node
from agent.nodes.router import router
from agent.state import AgentState
from data_pipeline.embedder.embedding_model import EmbeddingModel
from llm.ollama_client import OllamaClient
from retrieval.reranker import Reranker
from retrieval.retriever import Retriever


def _should_fallback_after_route(state: AgentState) -> str:
    """Decide if we should fallback after routing (off-topic)."""
    if state.get("should_fallback"):
        return "fallback"
    if not state.get("detected_domains"):
        return "retrieve"  # Search all domains
    return "retrieve"


def _grade_decision(state: AgentState) -> str:
    """Decide next step after grading documents."""
    graded = state.get("graded_documents", [])
    retry_count = state.get("retry_count", 0)

    if len(graded) >= 1:
        return "generate"
    elif retry_count < 2:
        return "retry"
    else:
        return "fallback"


def _hallucination_decision(state: AgentState) -> str:
    """Decide next step after hallucination check."""
    if state.get("is_grounded"):
        return "end"
    else:
        return "fallback"


def _increment_retry(state: AgentState) -> dict:
    """Increment the retry counter and use original query for retry."""
    return {
        "retry_count": state.get("retry_count", 0) + 1,
        "rewritten_query": state["query"],  # Fall back to original query
    }


def build_graph(
    collection: Collection,
    embedding_model: EmbeddingModel,
    ollama_client: OllamaClient,
    reranker: Reranker | None = None,
) -> StateGraph:
    """Build the LangGraph agent graph.

    Args:
        collection: Milvus collection for retrieval.
        embedding_model: Embedding model for query encoding.
        ollama_client: Ollama client for LLM inference.
        reranker: Optional reranker for retrieval.

    Returns:
        Compiled LangGraph StateGraph.
    """
    # Create retriever
    retriever_instance = Retriever(collection, embedding_model, reranker)

    # Bind dependencies to nodes
    analyze_node = partial(query_analyzer, llm=ollama_client)
    route_node = partial(router, llm=ollama_client)
    retrieve_node = partial(retriever_node, retriever=retriever_instance)
    grade_node = partial(grader, llm=ollama_client)
    generate_node = partial(generator, llm=ollama_client)
    hallucination_node = partial(hallucination_checker, llm=ollama_client)

    # Build graph
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("analyze", analyze_node)
    graph.add_node("route", route_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("generate", generate_node)
    graph.add_node("hallucination_check", hallucination_node)
    graph.add_node("fallback", fallback)
    graph.add_node("increment_retry", _increment_retry)

    # Define edges
    graph.set_entry_point("analyze")
    graph.add_edge("analyze", "route")

    graph.add_conditional_edges(
        "route",
        _should_fallback_after_route,
        {"retrieve": "retrieve", "fallback": "fallback"},
    )

    graph.add_edge("retrieve", "grade")

    graph.add_conditional_edges(
        "grade",
        _grade_decision,
        {"generate": "generate", "retry": "increment_retry", "fallback": "fallback"},
    )

    graph.add_edge("increment_retry", "retrieve")

    graph.add_edge("generate", "hallucination_check")

    graph.add_conditional_edges(
        "hallucination_check",
        _hallucination_decision,
        {"end": END, "fallback": "fallback"},
    )

    graph.add_edge("fallback", END)

    logger.info("Agent graph built successfully")
    return graph.compile()


def create_agent(
    collection: Collection,
    embedding_model: EmbeddingModel,
    ollama_client: OllamaClient | None = None,
    reranker: Reranker | None = None,
):
    """Create and return a compiled agent ready for invocation.

    Args:
        collection: Milvus collection.
        embedding_model: Embedding model.
        ollama_client: Ollama client (created if None).
        reranker: Optional reranker.

    Returns:
        Compiled LangGraph runnable.
    """
    if ollama_client is None:
        ollama_client = OllamaClient()

    return build_graph(collection, embedding_model, ollama_client, reranker)
