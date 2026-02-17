"""LangGraph agent definition: the core chatbot orchestration graph."""

from functools import partial

from langgraph.graph import END, StateGraph
from loguru import logger

from agent.nodes.fallback import fallback
from agent.nodes.generator import generator
from agent.nodes.grader import grader
from agent.nodes.quality_checker import quality_checker
from agent.nodes.query_analyzer import query_analyzer
from agent.nodes.retriever_node import retriever_node
from agent.nodes.router import router
from agent.state import AgentState
from data_pipeline.embedder.embedding_model import EmbeddingModel
from data_pipeline.store.vector_store import VectorStoreClient
from llm.ollama_client import OllamaClient
from retrieval.reranker import Reranker
from retrieval.retriever import Retriever


def _should_fallback_after_route(state: AgentState) -> str:
    """Decide if we should fallback after routing (off-topic)."""
    if state.get("should_fallback"):
        return "fallback"
    return "retrieve"


def _grade_decision(state: AgentState) -> str:
    """Decide next step after grading documents."""
    graded = state.get("graded_documents", [])
    retry_count = state.get("retry_count", 0)

    if len(graded) >= 1:
        return "generate"
    elif retry_count < 3:
        return "retry"
    else:
        return "fallback"


def _quality_decision(state: AgentState) -> str:
    """Decide next step after quality check.

    The quality checker can:
    - PASS: answer is good → end
    - REROUTE: wrong domain → go back to retrieve with new domain
    - REPHRASE: weak answer → go back to retrieve with rephrased query
    - FAIL: unrecoverable → fallback
    """
    action = state.get("quality_action", "fail")
    retry_count = state.get("retry_count", 0)

    if action == "pass":
        return "end"
    elif action in ("reroute", "rephrase") and retry_count < 3:
        return "retry"
    else:
        return "fallback"


def _increment_retry(state: AgentState) -> dict:
    """Increment the retry counter.

    The quality checker already sets the rewritten_query (for rephrase)
    or detected_domains (for reroute) in the state.
    """
    action = state.get("quality_action", "")
    retry_count = state.get("retry_count", 0) + 1

    result = {"retry_count": retry_count}

    # If no rephrase happened, fall back to original query
    if action != "rephrase":
        result["rewritten_query"] = state["query"]

    trace = state.get("reasoning_trace", [])
    trace.append(
        f"Retry #{retry_count} ({action}): "
        f"{state.get('quality_reasoning', '')[:100]}"
    )
    result["reasoning_trace"] = trace

    return result


def build_graph(
    store: VectorStoreClient,
    embedding_model: EmbeddingModel,
    ollama_client: OllamaClient,
    reranker: Reranker | None = None,
) -> StateGraph:
    """Build the LangGraph agent graph.

    Args:
        store: ChromaDB vector store client.
        embedding_model: Embedding model for query encoding.
        ollama_client: Ollama client for LLM inference.
        reranker: Optional reranker for retrieval.

    Returns:
        Compiled LangGraph StateGraph.
    """
    # Create retriever
    retriever_instance = Retriever(store, embedding_model, reranker)

    # Bind dependencies to nodes
    analyze_node = partial(query_analyzer, llm=ollama_client)
    route_node = partial(router, llm=ollama_client)
    retrieve_node = partial(retriever_node, retriever=retriever_instance)
    grade_node = partial(grader, llm=ollama_client)
    generate_node = partial(generator, llm=ollama_client)
    quality_node = partial(quality_checker, llm=ollama_client)

    # Build graph
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("analyze", analyze_node)
    graph.add_node("route", route_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("generate", generate_node)
    graph.add_node("quality_check", quality_node)
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

    graph.add_edge("generate", "quality_check")

    graph.add_conditional_edges(
        "quality_check",
        _quality_decision,
        {"end": END, "retry": "increment_retry", "fallback": "fallback"},
    )

    graph.add_edge("increment_retry", "retrieve")
    graph.add_edge("fallback", END)

    logger.info("Agent graph built successfully")
    return graph.compile()


def create_agent(
    store: VectorStoreClient,
    embedding_model: EmbeddingModel,
    ollama_client: OllamaClient | None = None,
    reranker: Reranker | None = None,
):
    """Create and return a compiled agent ready for invocation.

    Args:
        store: ChromaDB vector store client.
        embedding_model: Embedding model.
        ollama_client: Ollama client (created if None).
        reranker: Optional reranker.

    Returns:
        Compiled LangGraph runnable.
    """
    if ollama_client is None:
        ollama_client = OllamaClient()

    return build_graph(store, embedding_model, ollama_client, reranker)
