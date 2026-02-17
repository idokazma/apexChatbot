"""Main retrieval interface combining search, neighbor expansion, and reranking."""

from loguru import logger

from config.settings import settings
from data_pipeline.embedder.embedding_model import EmbeddingModel
from data_pipeline.store.vector_store import VectorStoreClient
from retrieval.hybrid_search import HybridSearcher
from retrieval.reranker import Reranker


class Retriever:
    """High-level retrieval interface: search -> expand neighbors -> rerank -> return."""

    def __init__(
        self,
        store: VectorStoreClient,
        embedding_model: EmbeddingModel,
        reranker: Reranker | None = None,
    ):
        self.store = store
        self.searcher = HybridSearcher(store, embedding_model)
        self.reranker = reranker

    def _expand_with_neighbors(self, results: list[dict]) -> list[dict]:
        """Expand each result with content from neighboring chunks.

        For each retrieved chunk, fetches the chunk before and after it from
        the same document and appends their content to the result. This ensures
        the LLM sees full context even when answers span chunk boundaries.
        """
        for doc in results:
            source_doc_id = doc.get("source_doc_id", "")
            chunk_index = doc.get("chunk_index", 0)

            if not source_doc_id:
                continue

            neighbors = self.store.get_neighbors(source_doc_id, chunk_index)

            neighbor_context = []
            if neighbors["prev"]:
                neighbor_context.append(neighbors["prev"]["content"])
            neighbor_context.append(doc.get("content", ""))
            if neighbors["next"]:
                neighbor_context.append(neighbors["next"]["content"])

            doc["content_expanded"] = "\n\n---\n\n".join(neighbor_context)

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

        # Expand with neighbor chunks
        results = self._expand_with_neighbors(results)

        logger.info(
            f"Retrieved {len(results)} docs (with neighbors) for: {query[:50]}... "
            f"(domains: {domain or domains or 'all'})"
        )
        return results
