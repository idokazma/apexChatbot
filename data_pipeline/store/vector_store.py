"""Vector database client for chunk storage and retrieval using ChromaDB."""

from pathlib import Path

import chromadb
from loguru import logger

from config.settings import settings


COLLECTION_NAME = settings.chromadb_collection


class VectorStoreClient:
    """Manages ChromaDB connection, collection creation, and data operations."""

    def __init__(self, persist_dir: str = str(Path(settings.data_dir) / "chromadb")):
        self.persist_dir = persist_dir
        self.collection_name = COLLECTION_NAME
        self._client: chromadb.ClientAPI | None = None
        self._collection = None

    def connect(self) -> None:
        """Initialize ChromaDB persistent client."""
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self.persist_dir)
        logger.info(f"Connected to ChromaDB at {self.persist_dir}")

    def disconnect(self) -> None:
        """No-op for ChromaDB (auto-persisted)."""
        pass

    @property
    def client(self) -> chromadb.ClientAPI:
        if self._client is None:
            self.connect()
        return self._client

    def create_collection(self, drop_existing: bool = False) -> None:
        """Create or get the collection."""
        if drop_existing:
            try:
                self.client.delete_collection(self.collection_name)
                logger.info(f"Dropped existing collection: {self.collection_name}")
            except Exception:
                pass

        self._collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "ip"},  # Inner Product
        )
        logger.info(f"Collection ready: {self.collection_name}")

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "ip"},
            )
        return self._collection

    def insert_chunks(self, embedded_chunks: list[dict], batch_size: int = 500) -> int:
        """Insert embedded chunks into ChromaDB.

        Args:
            embedded_chunks: List of dicts with 'chunk' (Chunk) and 'embedding' (list[float]).
            batch_size: Number of records per insert batch.

        Returns:
            Total number of inserted records.
        """
        total = 0

        for i in range(0, len(embedded_chunks), batch_size):
            batch = embedded_chunks[i : i + batch_size]

            ids = []
            embeddings = []
            documents = []
            metadatas = []

            for item in batch:
                chunk = item["chunk"]
                ids.append(chunk.metadata.chunk_id)
                embeddings.append(item["embedding"])
                documents.append(chunk.content_with_context[:10000])
                metadatas.append({
                    "content": chunk.content[:8000],
                    "domain": chunk.metadata.domain,
                    "source_url": chunk.metadata.source_url[:500],
                    "source_doc_title": chunk.metadata.source_doc_title[:250],
                    "source_doc_id": chunk.metadata.source_doc_id,
                    "section_path": chunk.metadata.section_path[:500],
                    "language": chunk.metadata.language,
                    "doc_type": chunk.metadata.doc_type,
                    "page_number": chunk.metadata.page_number or 0,
                    "chunk_index": chunk.metadata.chunk_index,
                    "source_file_path": chunk.metadata.source_file_path,
                    "keywords": ", ".join(chunk.metadata.keywords[:15]),
                    "summary": chunk.metadata.summary[:500],
                })

            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            total += len(batch)
            logger.debug(f"  Inserted batch {i // batch_size + 1}: {len(batch)} records")

        logger.info(f"Inserted {total} chunks into ChromaDB")
        return total

    def load_collection(self) -> None:
        """No-op for ChromaDB (always loaded)."""
        logger.info(f"Collection '{self.collection_name}' ready")

    def get_count(self) -> int:
        """Get the number of entities in the collection."""
        return self.collection.count()

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        domain_filter: str | None = None,
    ) -> list[dict]:
        """Search for similar chunks.

        Args:
            query_embedding: Query vector.
            top_k: Number of results to return.
            domain_filter: Optional domain to filter by.
        """
        where_filter = {"domain": domain_filter} if domain_filter else None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["embeddings", "documents", "metadatas", "distances"],
        )

        # Convert to flat list of dicts
        hits = []
        if results and results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                hit = {
                    "id": chunk_id,
                    "distance": results["distances"][0][i] if results["distances"] else 0,
                    "entity": {
                        "chunk_id": chunk_id,
                        "content_with_context": results["documents"][0][i] if results["documents"] else "",
                        **(results["metadatas"][0][i] if results["metadatas"] else {}),
                    },
                }
                hits.append(hit)

        return hits

    def get_neighbors(self, source_doc_id: str, chunk_index: int) -> dict:
        """Get the chunks immediately before and after a given chunk.

        Args:
            source_doc_id: Hash ID of the source document.
            chunk_index: Index of the target chunk.

        Returns:
            Dict with 'prev' and 'next' chunk content (or None).
        """
        neighbors = {"prev": None, "next": None}
        if not source_doc_id:
            return neighbors

        for offset, key in [(-1, "prev"), (1, "next")]:
            target_idx = chunk_index + offset
            if target_idx < 0:
                continue
            try:
                results = self.collection.get(
                    where={"$and": [
                        {"source_doc_id": source_doc_id},
                        {"chunk_index": target_idx},
                    ]},
                    include=["documents", "metadatas"],
                )
                if results and results["ids"]:
                    neighbors[key] = {
                        "content": results["metadatas"][0].get("content", "") if results["metadatas"] else "",
                        "content_with_context": results["documents"][0] if results["documents"] else "",
                    }
            except Exception:
                pass

        return neighbors
