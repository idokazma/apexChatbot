"""Application settings loaded from environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Anthropic
    anthropic_api_key: str = ""

    # OpenAI (baseline comparison)
    openai_api_key: str = ""

    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "gemma3:12b"

    # ChromaDB
    chromadb_collection: str = "harel_insurance"

    # Embedding
    embedding_model: str = "intfloat/multilingual-e5-large"
    embedding_dim: int = 1024

    # Retrieval
    top_k_retrieve: int = 10
    top_k_rerank: int = 5

    # Chunking
    max_chunk_tokens: int = 512
    chunk_overlap_tokens: int = 50

    # Paths
    data_dir: Path = Path("data")
    raw_data_dir: Path = Path("data/raw")
    parsed_data_dir: Path = Path("data/parsed")
    chunks_data_dir: Path = Path("data/chunks")

    # App
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
