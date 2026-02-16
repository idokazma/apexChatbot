"""Milvus vector database client for chunk storage and retrieval."""

from loguru import logger
from pymilvus import Collection, connections, utility

from config.settings import settings
from data_pipeline.store.schema import COLLECTION_NAME, get_collection_schema


class MilvusClient:
    """Manages Milvus connection, collection creation, and data operations."""

    def __init__(
        self,
        host: str = settings.milvus_host,
        port: int = settings.milvus_port,
    ):
        self.host = host
        self.port = port
        self.collection_name = COLLECTION_NAME
        self._collection: Collection | None = None

    def connect(self) -> None:
        """Establish connection to Milvus."""
        connections.connect("default", host=self.host, port=self.port)
        logger.info(f"Connected to Milvus at {self.host}:{self.port}")

    def disconnect(self) -> None:
        """Disconnect from Milvus."""
        connections.disconnect("default")

    def create_collection(self, drop_existing: bool = False) -> Collection:
        """Create the collection with schema and HNSW index."""
        if drop_existing and utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            logger.info(f"Dropped existing collection: {self.collection_name}")

        if utility.has_collection(self.collection_name):
            self._collection = Collection(self.collection_name)
            logger.info(f"Using existing collection: {self.collection_name}")
        else:
            schema = get_collection_schema()
            self._collection = Collection(
                name=self.collection_name,
                schema=schema,
            )
            logger.info(f"Created collection: {self.collection_name}")

            # Create HNSW index on embedding field
            index_params = {
                "metric_type": "IP",  # Inner Product (cosine with normalized vectors)
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 256},
            }
            self._collection.create_index("embedding", index_params)
            logger.info("Created HNSW index on embedding field")

        return self._collection

    @property
    def collection(self) -> Collection:
        if self._collection is None:
            self._collection = self.create_collection()
        return self._collection

    def insert_chunks(self, embedded_chunks: list[dict], batch_size: int = 500) -> int:
        """Insert embedded chunks into Milvus.

        Args:
            embedded_chunks: List of dicts with 'chunk' (Chunk) and 'embedding' (list[float]).
            batch_size: Number of records per insert batch.

        Returns:
            Total number of inserted records.
        """
        total = 0

        for i in range(0, len(embedded_chunks), batch_size):
            batch = embedded_chunks[i : i + batch_size]

            data = [
                [item["chunk"].metadata.chunk_id for item in batch],           # chunk_id
                [item["chunk"].content[:8000] for item in batch],               # content
                [item["chunk"].content_with_context[:10000] for item in batch],  # content_with_context
                [item["embedding"] for item in batch],                          # embedding
                [item["chunk"].metadata.domain for item in batch],              # domain
                [item["chunk"].metadata.source_url[:500] for item in batch],    # source_url
                [item["chunk"].metadata.source_doc_title[:250] for item in batch],  # source_doc_title
                [item["chunk"].metadata.section_path[:500] for item in batch],  # section_path
                [item["chunk"].metadata.language for item in batch],            # language
                [item["chunk"].metadata.doc_type for item in batch],            # doc_type
                [item["chunk"].metadata.page_number or 0 for item in batch],    # page_number
                [item["chunk"].metadata.chunk_index for item in batch],         # chunk_index
            ]

            self.collection.insert(data)
            total += len(batch)
            logger.debug(f"  Inserted batch {i // batch_size + 1}: {len(batch)} records")

        self.collection.flush()
        logger.info(f"Inserted {total} chunks into Milvus")
        return total

    def load_collection(self) -> None:
        """Load collection into memory for searching."""
        self.collection.load()
        logger.info(f"Collection '{self.collection_name}' loaded into memory")

    def get_count(self) -> int:
        """Get the number of entities in the collection."""
        return self.collection.num_entities
