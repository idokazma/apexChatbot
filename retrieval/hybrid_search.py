"""Hybrid search combining dense vector search with BM25 sparse retrieval."""

from loguru import logger
from pymilvus import Collection

from config.settings import settings
from data_pipeline.embedder.embedding_model import EmbeddingModel
from retrieval.query_processor import process_query


class HybridSearcher:
    """Performs hybrid search: dense (Milvus) + optional sparse (BM25) with RRF fusion."""

    def __init__(
        self,
        collection: Collection,
        embedding_model: EmbeddingModel,
    ):
        self.collection = collection
        self.embedding_model = embedding_model

    def search(
        self,
        query: str,
        top_k: int = settings.top_k_retrieve,
        domain_filter: str | None = None,
    ) -> list[dict]:
        """Search for relevant chunks using dense vector search.

        Args:
            query: User query text.
            top_k: Number of results to return.
            domain_filter: Optional domain name to filter results.

        Returns:
            List of result dicts with content, metadata, and score.
        """
        processed_query = process_query(query)
        query_embedding = self.embedding_model.embed_query(processed_query)

        # Build search params
        search_params = {
            "metric_type": "IP",
            "params": {"ef": 128},
        }

        # Build filter expression
        expr = None
        if domain_filter:
            expr = f'domain == "{domain_filter}"'

        # Output fields to retrieve
        output_fields = [
            "chunk_id",
            "content",
            "content_with_context",
            "domain",
            "source_url",
            "source_doc_title",
            "section_path",
            "language",
            "doc_type",
            "page_number",
            "chunk_index",
        ]

        results = self.collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=output_fields,
        )

        # Format results
        formatted = []
        for hits in results:
            for hit in hits:
                formatted.append({
                    "chunk_id": hit.entity.get("chunk_id"),
                    "content": hit.entity.get("content"),
                    "content_with_context": hit.entity.get("content_with_context"),
                    "domain": hit.entity.get("domain"),
                    "source_url": hit.entity.get("source_url"),
                    "source_doc_title": hit.entity.get("source_doc_title"),
                    "section_path": hit.entity.get("section_path"),
                    "language": hit.entity.get("language"),
                    "doc_type": hit.entity.get("doc_type"),
                    "page_number": hit.entity.get("page_number"),
                    "chunk_index": hit.entity.get("chunk_index"),
                    "score": hit.distance,
                })

        logger.debug(f"Dense search returned {len(formatted)} results for: {query[:50]}...")
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

        # Sort by score and take top_k
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]
