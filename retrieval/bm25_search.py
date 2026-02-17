"""BM25 sparse search over chunked documents stored in ChromaDB."""

import re

from loguru import logger
from rank_bm25 import BM25Okapi

from data_pipeline.store.vector_store import VectorStoreClient
from retrieval.query_processor import get_normalized_query


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for Hebrew/English."""
    text = text.lower()
    # Remove punctuation but keep Hebrew chars
    text = re.sub(r"[^\w\s\u0590-\u05FF]", " ", text)
    return text.split()


class BM25Index:
    """In-memory BM25 index built from ChromaDB collection."""

    def __init__(self):
        self._bm25: BM25Okapi | None = None
        self._docs: list[dict] = []

    @property
    def is_built(self) -> bool:
        return self._bm25 is not None

    def build_from_store(self, store: VectorStoreClient, domain: str | None = None) -> None:
        """Build BM25 index from all documents in the vector store.

        Args:
            store: ChromaDB vector store client.
            domain: Optional domain filter (builds index for one domain only).
        """
        where_filter = {"domain": domain} if domain else None

        # Fetch all documents from the collection
        results = store.collection.get(
            where=where_filter,
            include=["documents", "metadatas"],
        )

        if not results or not results["ids"]:
            logger.warning("No documents found for BM25 index")
            return

        self._docs = []
        corpus_tokens = []

        for i, chunk_id in enumerate(results["ids"]):
            content = results["documents"][i] if results["documents"] else ""
            metadata = results["metadatas"][i] if results["metadatas"] else {}

            doc = {
                "id": chunk_id,
                "content_with_context": content,
                **metadata,
            }
            self._docs.append(doc)
            corpus_tokens.append(_tokenize(content))

        self._bm25 = BM25Okapi(corpus_tokens)
        logger.info(f"BM25 index built with {len(self._docs)} documents")

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Search using BM25.

        Args:
            query: User query.
            top_k: Number of results.

        Returns:
            List of result dicts with BM25 score.
        """
        if not self.is_built:
            return []

        normalized_query = get_normalized_query(query)
        tokens = _tokenize(normalized_query)

        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = self._docs[idx]
                results.append({
                    "chunk_id": doc.get("chunk_id", doc["id"]),
                    "content": doc.get("content", ""),
                    "content_with_context": doc.get("content_with_context", ""),
                    "domain": doc.get("domain", ""),
                    "source_url": doc.get("source_url", ""),
                    "source_doc_title": doc.get("source_doc_title", ""),
                    "section_path": doc.get("section_path", ""),
                    "language": doc.get("language", ""),
                    "doc_type": doc.get("doc_type", ""),
                    "page_number": doc.get("page_number", 0),
                    "chunk_index": doc.get("chunk_index", 0),
                    "source_file_path": doc.get("source_file_path", ""),
                    "score": float(scores[idx]),
                })

        return results
