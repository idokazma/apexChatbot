"""Search module using ChromaDB vector store."""

from loguru import logger

from config.settings import settings
from data_pipeline.embedder.embedding_model import EmbeddingModel
from data_pipeline.store.milvus_client import VectorStoreClient
from retrieval.query_processor import process_query


class HybridSearcher:
    """Performs vector search via ChromaDB with optional domain filtering."""

    def __init__(
        self,
        store: VectorStoreClient,
        embedding_model: EmbeddingModel,
    ):
        self.store = store
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

        results = self.store.search(
            query_embedding=query_embedding.tolist(),
            top_k=top_k,
            domain_filter=domain_filter,
        )

        # Format results
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

        logger.debug(f"Search returned {len(formatted)} results for: {query[:50]}...")
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
