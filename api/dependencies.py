"""FastAPI dependency injection: shared resources initialized once."""

from dataclasses import dataclass, field

from pymilvus import Collection

from data_pipeline.embedder.embedding_model import EmbeddingModel
from data_pipeline.store.milvus_client import MilvusClient
from llm.ollama_client import OllamaClient
from retrieval.reranker import Reranker


@dataclass
class AppResources:
    """Container for shared application resources."""

    milvus_client: MilvusClient | None = None
    collection: Collection | None = None
    embedding_model: EmbeddingModel | None = None
    ollama_client: OllamaClient | None = None
    reranker: Reranker | None = None
    agent: object | None = None  # Compiled LangGraph
    _initialized: bool = False

    def initialize(self) -> None:
        """Initialize all resources. Called once at startup."""
        if self._initialized:
            return

        from agent.graph import create_agent

        # Milvus
        self.milvus_client = MilvusClient()
        self.milvus_client.connect()
        self.collection = self.milvus_client.create_collection()
        self.milvus_client.load_collection()

        # Embedding model
        self.embedding_model = EmbeddingModel()

        # LLM
        self.ollama_client = OllamaClient()

        # Reranker
        self.reranker = Reranker()

        # Agent
        self.agent = create_agent(
            collection=self.collection,
            embedding_model=self.embedding_model,
            ollama_client=self.ollama_client,
            reranker=self.reranker,
        )

        self._initialized = True

    def shutdown(self) -> None:
        """Clean up resources."""
        if self.milvus_client:
            self.milvus_client.disconnect()


# Global singleton
resources = AppResources()
