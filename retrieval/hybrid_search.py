"""Hybrid search: dense vector (ChromaDB) + sparse BM25 with Reciprocal Rank Fusion."""

from loguru import logger

from config.settings import settings
from data_pipeline.embedder.embedding_model import EmbeddingModel
from data_pipeline.store.vector_store import VectorStoreClient
from retrieval.bm25_search import BM25Index
from retrieval.query_processor import process_query


def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    """Merge multiple ranked result lists using Reciprocal Rank Fusion (RRF).

    Args:
        result_lists: List of ranked result lists (each from a different retriever).
        k: RRF constant (default 60, standard in literature).

    Returns:
        Merged and re-ranked results.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            doc_id = doc.get("chunk_id", doc.get("id", str(rank)))
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            if doc_id not in doc_map:
                doc_map[doc_id] = doc

    # Sort by RRF score
    ranked_ids = sorted(scores, key=scores.get, reverse=True)
    merged = []
    for doc_id in ranked_ids:
        doc = doc_map[doc_id]
        doc["rrf_score"] = scores[doc_id]
        merged.append(doc)

    return merged


class HybridSearcher:
    """Performs hybrid search: dense (ChromaDB) + sparse (BM25) with RRF fusion."""

    def __init__(
        self,
        store: VectorStoreClient,
        embedding_model: EmbeddingModel,
    ):
        self.store = store
        self.embedding_model = embedding_model
        self._bm25_index = BM25Index()

    def _ensure_bm25(self) -> None:
        """Lazily build the BM25 index on first search."""
        if not self._bm25_index.is_built:
            logger.info("Building BM25 index (first search)...")
            self._bm25_index.build_from_store(self.store)

    def search(
        self,
        query: str,
        top_k: int = settings.top_k_retrieve,
        domain_filter: str | None = None,
    ) -> list[dict]:
        """Hybrid search combining dense vectors and BM25.

        Args:
            query: User query text.
            top_k: Number of results to return.
            domain_filter: Optional domain name to filter results.

        Returns:
            List of result dicts with content, metadata, and score.
        """
        processed_query = process_query(query)
        query_embedding = self.embedding_model.embed_query(processed_query)

        # Dense search (ChromaDB)
        dense_results = self.store.search(
            query_embedding=query_embedding.tolist(),
            top_k=top_k,
            domain_filter=domain_filter,
        )
        dense_formatted = self._format_chromadb_results(dense_results)

        # Sparse search (BM25)
        self._ensure_bm25()
        sparse_results = self._bm25_index.search(query, top_k=top_k)

        # Apply domain filter to BM25 results if needed
        if domain_filter:
            sparse_results = [r for r in sparse_results if r.get("domain") == domain_filter]

        # Fuse with RRF
        merged = reciprocal_rank_fusion([dense_formatted, sparse_results])

        logger.debug(
            f"Hybrid search: {len(dense_formatted)} dense + {len(sparse_results)} sparse "
            f"→ {len(merged)} merged for: {query[:50]}..."
        )
        return merged[:top_k]

    def _format_chromadb_results(self, results: list[dict]) -> list[dict]:
        """Convert ChromaDB result format to standard format."""
        formatted = []
        for hit in results:
            entity = hit["entity"]
            formatted.append({
                "chunk_id": entity.get("chunk_id", hit["id"]),
                "content": entity.get("content", ""),
                "content_with_context": entity.get("content_with_context", ""),
                "domain": entity.get("domain", ""),
                "source_url": entity.get("source_url", ""),
                "source_doc_title": entity.get("source_doc_title", ""),
                "section_path": entity.get("section_path", ""),
                "language": entity.get("language", ""),
                "doc_type": entity.get("doc_type", ""),
                "page_number": entity.get("page_number", 0),
                "chunk_index": entity.get("chunk_index", 0),
                "score": hit.get("distance", 0),
            })
        return formatted

    def multi_domain_search(
        self,
        query: str,
        domains: list[str],
        top_k: int = settings.top_k_retrieve,
    ) -> list[dict]:
        """Search across multiple specific domains and merge results."""
        all_results = []

        for domain in domains:
            results = self.search(query, top_k=top_k, domain_filter=domain)
            all_results.extend(results)

        # Sort by RRF score (or original score) and take top_k
        all_results.sort(
            key=lambda x: x.get("rrf_score", x.get("score", 0)),
            reverse=True,
        )
        return all_results[:top_k]
