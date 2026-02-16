"""Main retrieval interface combining search and reranking."""

from loguru import logger

from config.settings import settings
from data_pipeline.embedder.embedding_model import EmbeddingModel
from data_pipeline.store.milvus_client import VectorStoreClient
from retrieval.hybrid_search import HybridSearcher
from retrieval.reranker import Reranker


class Retriever:
    """High-level retrieval interface: search -> rerank -> return top results."""

    def __init__(
        self,
        store: VectorStoreClient,
        embedding_model: EmbeddingModel,
        reranker: Reranker | None = None,
    ):
        self.searcher = HybridSearcher(store, embedding_model)
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        domain: str | None = None,
        domains: list[str] | None = None,
        top_k_search: int = settings.top_k_retrieve,
        top_k_final: int = settings.top_k_rerank,
    ) -> list[dict]:
        """Retrieve and rerank documents for a query.

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

        logger.info(
            f"Retrieved {len(results)} docs for: {query[:50]}... "
            f"(domains: {domain or domains or 'all'})"
        )
        return results
