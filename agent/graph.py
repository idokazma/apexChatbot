"""LangGraph agent definition: the core chatbot orchestration graph.

Supports three retrieval modes:
  - "rag"      : classic vector search → route → retrieve → grade → generate
  - "agentic"  : hierarchy navigation → navigate → generate
  - "combined" : both RAG + navigator in parallel → merge → generate
"""

from functools import partial
from pathlib import Path

from langgraph.graph import END, StateGraph
from loguru import logger

from agent.nodes.combined_retriever_node import combined_retriever_node
from agent.nodes.fallback import fallback
from agent.nodes.generator import generator
from agent.nodes.grader import grader
from agent.nodes.navigate_node import navigate_node
from agent.nodes.quality_checker import quality_checker
from agent.nodes.query_analyzer import query_analyzer
from agent.nodes.retriever_node import retriever_node
from agent.nodes.router import router
from agent.state import AgentState
from data_pipeline.embedder.embedding_model import EmbeddingModel
from data_pipeline.store.vector_store import VectorStoreClient
from llm.ollama_client import OllamaClient
from llm.trace import set_current_node
from retrieval.navigator.hierarchy_store import HierarchyStore
from retrieval.navigator.navigator import Navigator
from retrieval.reranker import Reranker
from retrieval.retriever import Retriever


# ── Node tracing wrapper ──────────────────────────────────────────


def _traced_node(name: str, fn):
    """Wrap a node function to set the active trace node name before execution."""
    def wrapper(state):
        set_current_node(name)
        return fn(state)
    return wrapper


# ── Conditional edge helpers ───────────────────────────────────────


def _should_fallback_after_route(state: AgentState) -> str:
    """Decide if we should fallback after routing (off-topic)."""
    if state.get("should_fallback"):
        return "fallback"
    return "retrieve"


def _navigate_decision(state: AgentState) -> str:
    """Decide next step after navigation (agentic/combined)."""
    if state.get("should_fallback"):
        return "fallback"
    docs = state.get("retrieved_documents", [])
    if docs:
        return "generate"
    return "fallback"


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


# ── Graph builders ─────────────────────────────────────────────────


def build_rag_graph(
    store: VectorStoreClient,
    embedding_model: EmbeddingModel,
    ollama_client: OllamaClient,
    reranker: Reranker | None = None,
) -> StateGraph:
    """Build the classic RAG agent graph.

    Flow: analyze → route → retrieve → grade → generate → quality_check
    """
    retriever_instance = Retriever(store, embedding_model, reranker)

    analyze_node_fn = partial(query_analyzer, llm=ollama_client)
    route_node = partial(router, llm=ollama_client)
    retrieve_node = partial(retriever_node, retriever=retriever_instance)
    grade_node = partial(grader, llm=ollama_client)
    generate_node = partial(generator, llm=ollama_client)
    quality_node = partial(quality_checker, llm=ollama_client)

    graph = StateGraph(AgentState)

    graph.add_node("analyze", _traced_node("analyze", analyze_node_fn))
    graph.add_node("route", _traced_node("route", route_node))
    graph.add_node("retrieve", _traced_node("retrieve", retrieve_node))
    graph.add_node("grade", _traced_node("grade", grade_node))
    graph.add_node("generate", _traced_node("generate", generate_node))
    graph.add_node("quality_check", _traced_node("quality_check", quality_node))
    graph.add_node("fallback", _traced_node("fallback", fallback))
    graph.add_node("increment_retry", _traced_node("increment_retry", _increment_retry))

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

    logger.info("RAG agent graph built successfully")
    return graph.compile()


def build_agentic_graph(
    ollama_client: OllamaClient,
    hierarchy_store: HierarchyStore,
) -> StateGraph:
    """Build the agentic navigation graph.

    Flow: analyze → navigate → generate → quality_check → [end/retry/fallback]
    """
    navigator_instance = Navigator(hierarchy_store, ollama_client)

    analyze_node_fn = partial(query_analyzer, llm=ollama_client)
    nav_node = partial(navigate_node, navigator=navigator_instance)
    generate_node = partial(generator, llm=ollama_client)
    quality_node = partial(quality_checker, llm=ollama_client)

    graph = StateGraph(AgentState)

    graph.add_node("analyze", _traced_node("analyze", analyze_node_fn))
    graph.add_node("navigate", _traced_node("navigate", nav_node))
    graph.add_node("generate", _traced_node("generate", generate_node))
    graph.add_node("quality_check", _traced_node("quality_check", quality_node))
    graph.add_node("fallback", _traced_node("fallback", fallback))
    graph.add_node("increment_retry", _traced_node("increment_retry", _increment_retry))

    graph.set_entry_point("analyze")
    graph.add_edge("analyze", "navigate")

    graph.add_conditional_edges(
        "navigate",
        _navigate_decision,
        {"generate": "generate", "fallback": "fallback"},
    )

    graph.add_edge("generate", "quality_check")

    graph.add_conditional_edges(
        "quality_check",
        _quality_decision,
        {"end": END, "retry": "increment_retry", "fallback": "fallback"},
    )

    graph.add_edge("increment_retry", "navigate")
    graph.add_edge("fallback", END)

    logger.info("Agentic navigation graph built successfully")
    return graph.compile()


def build_combined_graph(
    store: VectorStoreClient,
    embedding_model: EmbeddingModel,
    ollama_client: OllamaClient,
    hierarchy_store: HierarchyStore,
    reranker: Reranker | None = None,
) -> StateGraph:
    """Build the combined RAG + agentic navigation graph.

    Flow: analyze → combined_retrieve → generate → quality_check → [end/retry/fallback]

    The combined_retrieve node runs both the classic RAG retriever and the
    agentic navigator, then merges and deduplicates the results.
    """
    retriever_instance = Retriever(store, embedding_model, reranker)
    navigator_instance = Navigator(hierarchy_store, ollama_client)

    analyze_node_fn = partial(query_analyzer, llm=ollama_client)
    combined_node = partial(
        combined_retriever_node,
        retriever=retriever_instance,
        navigator=navigator_instance,
    )
    generate_node = partial(generator, llm=ollama_client)
    quality_node = partial(quality_checker, llm=ollama_client)

    graph = StateGraph(AgentState)

    graph.add_node("analyze", _traced_node("analyze", analyze_node_fn))
    graph.add_node("combined_retrieve", _traced_node("combined_retrieve", combined_node))
    graph.add_node("generate", _traced_node("generate", generate_node))
    graph.add_node("quality_check", _traced_node("quality_check", quality_node))
    graph.add_node("fallback", _traced_node("fallback", fallback))
    graph.add_node("increment_retry", _traced_node("increment_retry", _increment_retry))

    graph.set_entry_point("analyze")
    graph.add_edge("analyze", "combined_retrieve")

    graph.add_conditional_edges(
        "combined_retrieve",
        _navigate_decision,  # reuse: checks retrieved_documents + should_fallback
        {"generate": "generate", "fallback": "fallback"},
    )

    graph.add_edge("generate", "quality_check")

    graph.add_conditional_edges(
        "quality_check",
        _quality_decision,
        {"end": END, "retry": "increment_retry", "fallback": "fallback"},
    )

    graph.add_edge("increment_retry", "combined_retrieve")
    graph.add_edge("fallback", END)

    logger.info("Combined (RAG + agentic) graph built successfully")
    return graph.compile()


# ── Factory functions ──────────────────────────────────────────────


def create_agent_for_mode(
    mode: str,
    store: VectorStoreClient | None = None,
    embedding_model: EmbeddingModel | None = None,
    ollama_client: OllamaClient | None = None,
    reranker: Reranker | None = None,
    hierarchy_dir: Path = Path("data/hierarchy"),
):
    """Create an agent for the given retrieval mode.

    Args:
        mode: One of "rag", "agentic", or "combined".
        store: ChromaDB vector store (required for "rag" and "combined").
        embedding_model: Embedding model (required for "rag" and "combined").
        ollama_client: Ollama client (created if None).
        reranker: Optional reranker (used by "rag" and "combined").
        hierarchy_dir: Path to hierarchy data (required for "agentic" and "combined").

    Returns:
        Compiled LangGraph runnable.

    Raises:
        ValueError: If mode is unknown.
        FileNotFoundError: If hierarchy data is missing for agentic/combined modes.
    """
    if ollama_client is None:
        ollama_client = OllamaClient()

    if mode == "rag":
        if store is None or embedding_model is None:
            raise ValueError("RAG mode requires store and embedding_model")
        return build_rag_graph(store, embedding_model, ollama_client, reranker)

    elif mode == "agentic":
        hierarchy_store = HierarchyStore(hierarchy_dir)
        if not hierarchy_store.is_ready():
            raise FileNotFoundError(
                f"Hierarchy data not found at {hierarchy_dir}. "
                "Run: python -m data_pipeline.hierarchy.build_hierarchy"
            )
        return build_agentic_graph(ollama_client, hierarchy_store)

    elif mode == "combined":
        if store is None or embedding_model is None:
            raise ValueError("Combined mode requires store and embedding_model")
        hierarchy_store = HierarchyStore(hierarchy_dir)
        if not hierarchy_store.is_ready():
            raise FileNotFoundError(
                f"Hierarchy data not found at {hierarchy_dir}. "
                "Run: python -m data_pipeline.hierarchy.build_hierarchy"
            )
        return build_combined_graph(
            store, embedding_model, ollama_client, hierarchy_store, reranker
        )

    else:
        raise ValueError(
            f"Unknown retrieval mode: '{mode}'. "
            "Use 'rag', 'agentic', or 'combined'."
        )


def create_agent(
    store: VectorStoreClient,
    embedding_model: EmbeddingModel,
    ollama_client: OllamaClient | None = None,
    reranker: Reranker | None = None,
):
    """Create a classic RAG agent (backward-compatible).

    For new code, prefer create_agent_for_mode().
    """
    if ollama_client is None:
        ollama_client = OllamaClient()

    return build_rag_graph(store, embedding_model, ollama_client, reranker)


def create_agentic_agent(
    ollama_client: OllamaClient | None = None,
    hierarchy_dir: Path = Path("data/hierarchy"),
):
    """Create an agentic navigation agent (backward-compatible).

    For new code, prefer create_agent_for_mode().
    """
    return create_agent_for_mode(
        mode="agentic",
        ollama_client=ollama_client,
        hierarchy_dir=hierarchy_dir,
    )
