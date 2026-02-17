"""Sequential retrieval cascade: BM25 broad retrieval → dense reranking."""

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
    """Sequential retrieval cascade: BM25 broad retrieval → dense reranking."""

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
        """Sequential cascade: BM25 → dense reranking.

        Stage 1: BM25 keyword search for broad candidate retrieval
        Stage 2: Dense embedding scoring for semantic reranking
        Fallback: Direct dense search if BM25 yields too few candidates
        """
        processed_query = process_query(query)

        # Stage 1: BM25 broad retrieval
        self._ensure_bm25()
        bm25_candidates = self._bm25_index.search(query, top_k=top_k * 3)

        if domain_filter:
            bm25_candidates = [r for r in bm25_candidates if r.get("domain") == domain_filter]

        # Stage 2: Dense reranking of BM25 candidates
        min_candidates = 3
        if len(bm25_candidates) >= min_candidates:
            # Rerank BM25 results using dense embeddings
            query_embedding = self.embedding_model.embed_query(processed_query)
            reranked = self._dense_rerank(bm25_candidates, query_embedding)

            logger.debug(
                f"Cascade search: {len(bm25_candidates)} BM25 candidates → "
                f"{len(reranked[:top_k])} after dense reranking for: {query[:50]}..."
            )
            return reranked[:top_k]

        # Fallback: BM25 found too few, use direct dense search + RRF merge
        logger.debug(f"BM25 found only {len(bm25_candidates)} candidates, using fallback dense search")
        query_embedding = self.embedding_model.embed_query(processed_query)
        dense_results = self.store.search(
            query_embedding=query_embedding.tolist(),
            top_k=top_k,
            domain_filter=domain_filter,
        )
        dense_formatted = self._format_chromadb_results(dense_results)

        # Merge sparse + dense via RRF as fallback
        merged = reciprocal_rank_fusion([bm25_candidates, dense_formatted])

        logger.debug(
            f"Fallback hybrid: {len(bm25_candidates)} BM25 + {len(dense_formatted)} dense "
            f"→ {len(merged[:top_k])} merged for: {query[:50]}..."
        )
        return merged[:top_k]

    def _dense_rerank(self, candidates: list[dict], query_embedding: list[float]) -> list[dict]:
        """Rerank candidates using cosine similarity with query embedding."""
        import numpy as np

        if not candidates:
            return []

        # Score each candidate by embedding its content and computing similarity
        scored = []
        for candidate in candidates:
            content = candidate.get("content", "")
            if not content:
                scored.append((candidate, 0.0))
                continue

            # Embed the candidate content and compute cosine similarity
            content_embedding = self.embedding_model.embed_query(content[:512])
            similarity = np.dot(query_embedding, content_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(content_embedding) + 1e-8
            )
            candidate["dense_score"] = float(similarity)
            scored.append((candidate, float(similarity)))

        # Sort by dense similarity
        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored]

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
                "source_file_path": entity.get("source_file_path", ""),
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
