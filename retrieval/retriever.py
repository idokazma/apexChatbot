"""Main retrieval interface combining search, neighbor expansion, and reranking."""

from loguru import logger

from config.settings import settings
from data_pipeline.embedder.embedding_model import EmbeddingModel
from data_pipeline.store.vector_store import VectorStoreClient
from retrieval.hybrid_search import HybridSearcher
from retrieval.reranker import Reranker


class Retriever:
    """High-level retrieval interface: search -> rerank -> expand full document -> return."""

    def __init__(
        self,
        store: VectorStoreClient,
        embedding_model: EmbeddingModel,
        reranker: Reranker | None = None,
    ):
        self.store = store
        self.searcher = HybridSearcher(store, embedding_model)
        self.reranker = reranker

    def _expand_with_full_document(self, results: list[dict]) -> list[dict]:
        """Expand each result with the full document context.

        For each retrieved chunk, loads all sibling chunks from the same
        source document and concatenates them in order. This gives the LLM
        the complete document context, not just a narrow window.
        Caches per source_doc_id to avoid redundant lookups when multiple
        chunks come from the same document.
        """
        doc_cache: dict[str, list[dict]] = {}

        for doc in results:
            source_doc_id = doc.get("source_doc_id", "")
            if not source_doc_id:
                continue

            if source_doc_id not in doc_cache:
                doc_cache[source_doc_id] = self.store.get_document_chunks(source_doc_id)

            all_chunks = doc_cache[source_doc_id]
            if not all_chunks:
                continue

            doc["content_expanded"] = "\n\n---\n\n".join(
                c["content"] for c in all_chunks if c.get("content")
            )

        return results

    def retrieve(
        self,
        query: str,
        domain: str | None = None,
        domains: list[str] | None = None,
        top_k_search: int = settings.top_k_retrieve,
        top_k_final: int = settings.top_k_rerank,
    ) -> list[dict]:
        """Retrieve, expand neighbors, and rerank documents for a query.

        Args:
            query: User query.
            domain: Single domain to filter by.
            domains: Multiple domains to search across.
            top_k_search: Number of initial retrieval results.
            top_k_final: Number of results after reranking.

        Returns:
            List of top-k result dicts with content and metadata.
        """
        # Search
        if domains and len(domains) > 1:
            results = self.searcher.multi_domain_search(query, domains, top_k=top_k_search)
        else:
            filter_domain = domain or (domains[0] if domains else None)
            results = self.searcher.search(query, top_k=top_k_search, domain_filter=filter_domain)

        if not results:
            logger.warning(f"No results found for query: {query[:80]}")
            return []

        # Rerank
        if self.reranker and len(results) > top_k_final:
            results = self.reranker.rerank(query, results, top_k=top_k_final)
        else:
            results = results[:top_k_final]

        # Expand with full document context
        results = self._expand_with_full_document(results)

        logger.info(
            f"Retrieved {len(results)} docs (with full document context) for: {query[:50]}... "
            f"(domains: {domain or domains or 'all'})"
        )
        return results
