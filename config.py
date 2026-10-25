"""
RFP Intelligence Platform – Configuration Module
Centralizes all settings; reads from .env / environment variables.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM ─────────────────────────────────────────────────────────────────
    llm_provider: str = Field("ollama", validation_alias="LLM_PROVIDER")
    llm_model: str = Field("llama3.2", validation_alias="LLM_MODEL")
    ollama_base_url: str = Field("http://localhost:11434", validation_alias="OLLAMA_BASE_URL")

    # ── Embeddings ───────────────────────────────────────────────────────────
    embedding_model: str = Field("nomic-embed-text", validation_alias="EMBEDDING_MODEL")
    embedding_dim: int = Field(768, validation_alias="EMBEDDING_DIM")

    # ── MongoDB ──────────────────────────────────────────────────────────────
    mongodb_uri: str = Field("mongodb://localhost:27017", validation_alias="MONGODB_URI")
    mongodb_db_name: str = Field("rfp_intelligence", validation_alias="MONGODB_DB_NAME")
    mongodb_collection_chunks: str = Field("document_chunks", validation_alias="MONGODB_COLLECTION_CHUNKS")
    mongodb_collection_bids: str = Field("bids", validation_alias="MONGODB_COLLECTION_BIDS")
    mongodb_vector_index: str = Field("vector_index", validation_alias="MONGODB_VECTOR_INDEX")

    # ── BM25 ─────────────────────────────────────────────────────────────────
    bm25_index_dir: Path = Field(Path("./data/bm25_indices"), validation_alias="BM25_INDEX_DIR")

    # ── Retrieval ────────────────────────────────────────────────────────────
    top_k_retrieval: int = Field(10, validation_alias="TOP_K_RETRIEVAL")
    top_k_rerank: int = Field(5, validation_alias="TOP_K_RERANK")
    hybrid_alpha: float = Field(0.6, validation_alias="HYBRID_ALPHA")

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunk_size: int = Field(512, validation_alias="CHUNK_SIZE")
    chunk_overlap: int = Field(64, validation_alias="CHUNK_OVERLAP")

    # ── API ──────────────────────────────────────────────────────────────────
    api_host: str = Field("0.0.0.0", validation_alias="API_HOST")
    api_port: int = Field(8000, validation_alias="API_PORT")

    # ── Logging ──────────────────────────────────────────────────────────────
    log_level: str = Field("INFO", validation_alias="LOG_LEVEL")
    log_dir: Path = Field(Path("./logs"), validation_alias="LOG_DIR")

    # ── Agents ───────────────────────────────────────────────────────────────
    max_retries: int = Field(3, validation_alias="MAX_RETRIES")
    max_agent_iterations: int = Field(10, validation_alias="MAX_AGENT_ITERATIONS")

    # ── LangSmith (optional) ─────────────────────────────────────────────────
    langchain_tracing_v2: Optional[str] = Field(None, validation_alias="LANGCHAIN_TRACING_V2")
    langchain_api_key: Optional[str] = Field(None, validation_alias="LANGCHAIN_API_KEY")
    langchain_project: Optional[str] = Field(None, validation_alias="LANGCHAIN_PROJECT")

    def ensure_dirs(self) -> None:
        """Create required directories if they don't exist."""
        for d in [self.bm25_index_dir, self.log_dir, Path("./data/bids"), Path("./output")]:
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
