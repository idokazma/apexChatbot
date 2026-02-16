"""FastAPI dependency injection: shared resources initialized once."""

from dataclasses import dataclass

from data_pipeline.embedder.embedding_model import EmbeddingModel
from data_pipeline.store.vector_store import VectorStoreClient
from llm.ollama_client import OllamaClient
from retrieval.reranker import Reranker


@dataclass
class AppResources:
    """Container for shared application resources."""

    store: VectorStoreClient | None = None
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

        # Vector store (ChromaDB)
        self.store = VectorStoreClient()
        self.store.connect()

        # Embedding model
        self.embedding_model = EmbeddingModel()

        # LLM
        self.ollama_client = OllamaClient()

        # Reranker
        self.reranker = Reranker()

        # Agent
        self.agent = create_agent(
            store=self.store,
            embedding_model=self.embedding_model,
            ollama_client=self.ollama_client,
            reranker=self.reranker,
        )

        self._initialized = True

    def shutdown(self) -> None:
        """Clean up resources."""
        if self.store:
            self.store.disconnect()


# Global singleton
resources = AppResources()
