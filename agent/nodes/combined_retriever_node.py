"""Combined retriever node: runs both RAG vector search and agentic navigation,
then merges and deduplicates the results.

Used by the 'combined' retrieval mode to get the best of both approaches:
- RAG finds semantically similar chunks (good for unexpected matches)
- Navigator finds structurally relevant chunks (good for precise drill-down)
"""

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from loguru import logger

from agent.state import AgentState
from retrieval.navigator.navigator import Navigator
from retrieval.retriever import Retriever

NAV_TIMEOUT_S = 30  # max seconds to wait for navigator


def combined_retriever_node(
    state: AgentState,
    retriever: Retriever,
    navigator: Navigator,
) -> dict:
    """Retrieve via both RAG and agentic navigation in parallel, merge results.

    Deduplicates by chunk_id, keeping the first occurrence (navigator
    results come first since they're more structurally targeted).
    Navigator is capped at 30s — if it's slow, RAG results are used alone.
    """
    query = state.get("rewritten_query") or state["query"]
    domains = state.get("detected_domains", [])
    language = state.get("detected_language", "he")

    nav_docs: list[dict] = []
    nav_path: dict = {}
    rag_docs: list[dict] = []

    # ── Run navigator and RAG in parallel ─────────────────────────
    def _run_navigator():
        return navigator.navigate(query, language)

    def _run_rag():
        domain = domains[0] if len(domains) == 1 else None
        return retriever.retrieve(query, domain=domain)

    t0 = time.time()

    with ThreadPoolExecutor(max_workers=2) as pool:
        nav_future = pool.submit(_run_navigator)
        rag_future = pool.submit(_run_rag)

        # RAG is fast — wait for it without a tight timeout
        try:
            rag_docs = rag_future.result(timeout=60)
            logger.info(f"RAG returned {len(rag_docs)} chunks in {time.time() - t0:.1f}s")
        except Exception as e:
            logger.warning(f"RAG retriever failed: {e}")

        # Navigator gets a 30s cap
        try:
            nav_result = nav_future.result(timeout=NAV_TIMEOUT_S)
            nav_docs = nav_result.get("retrieved_documents", [])
            nav_path = nav_result.get("navigation_path", {})
            logger.info(f"Navigator returned {len(nav_docs)} chunks in {time.time() - t0:.1f}s")
        except FuturesTimeoutError:
            logger.warning(f"Navigator timed out after {NAV_TIMEOUT_S}s, using RAG results only")
            nav_path = {"trace": [f"Navigator timed out after {NAV_TIMEOUT_S}s"]}
        except Exception as e:
            logger.warning(f"Navigator failed, continuing with RAG only: {e}")
            nav_path = {"trace": [f"Navigator error: {e}"]}

    # ── Merge and deduplicate ──────────────────────────────────────
    seen: set[str] = set()
    merged: list[dict] = []

    # Navigator results first (structurally targeted)
    for doc in nav_docs:
        cid = doc.get("chunk_id", "")
        if cid and cid not in seen:
            seen.add(cid)
            doc["_source"] = "navigator"
            merged.append(doc)

    # RAG results second (semantically similar)
    for doc in rag_docs:
        cid = doc.get("chunk_id") or doc.get("entity", {}).get("chunk_id", "")
        if cid and cid not in seen:
            seen.add(cid)
            # Normalize RAG hit format to match navigator output
            if "entity" in doc:
                entity = doc["entity"]
                normalized = {
                    "chunk_id": cid,
                    "content": entity.get("content", ""),
                    "content_with_context": entity.get("content_with_context", ""),
                    "source_url": entity.get("source_url", ""),
                    "source_doc_title": entity.get("source_doc_title", ""),
                    "source_doc_id": entity.get("source_doc_id", ""),
                    "domain": entity.get("domain", ""),
                    "section_path": entity.get("section_path", ""),
                    "page_number": entity.get("page_number"),
                    "source_file_path": entity.get("source_file_path", ""),
                    "language": entity.get("language", "he"),
                    "doc_type": entity.get("doc_type", ""),
                    "chunk_index": entity.get("chunk_index", 0),
                    "_source": "rag",
                }
                merged.append(normalized)
            else:
                doc["_source"] = "rag"
                merged.append(doc)

    should_fallback = len(merged) == 0

    trace = state.get("reasoning_trace", [])
    trace.append(
        f"Combined retriever: {len(nav_docs)} from navigator + "
        f"{len(rag_docs)} from RAG = {len(merged)} merged (deduplicated)"
    )
    trace.extend(nav_path.get("trace", []))

    elapsed = time.time() - t0
    logger.info(
        f"Combined: {len(nav_docs)} nav + {len(rag_docs)} rag = {len(merged)} merged in {elapsed:.1f}s"
    )

    return {
        "retrieved_documents": merged,
        "graded_documents": merged,  # both paths pre-filter
        "navigation_path": nav_path,
        "should_fallback": should_fallback,
        "reasoning_trace": trace,
    }
