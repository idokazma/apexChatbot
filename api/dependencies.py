"""FastAPI dependency injection: shared resources initialized once."""

from dataclasses import dataclass

from loguru import logger

from config.settings import settings
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

        from agent.graph import create_agent_for_mode

        mode = settings.retrieval_mode

        # LLM
        self.ollama_client = OllamaClient()

        # Vector store + embeddings (needed for "rag" and "combined" modes)
        if mode in ("rag", "combined"):
            self.store = VectorStoreClient()
            self.store.connect()
            self.embedding_model = EmbeddingModel()
            self.reranker = Reranker()

        # Agent
        self.agent = create_agent_for_mode(
            mode=mode,
            store=self.store,
            embedding_model=self.embedding_model,
            ollama_client=self.ollama_client,
            reranker=self.reranker,
            hierarchy_dir=settings.hierarchy_dir,
        )

        logger.info(f"Agent initialized in '{mode}' retrieval mode")
        self._initialized = True

    def shutdown(self) -> None:
        """Clean up resources."""
        if self.store:
            self.store.disconnect()


# Global singleton
resources = AppResources()
