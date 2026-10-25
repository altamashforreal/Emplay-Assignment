"""
Embedder – generates text embeddings via Ollama (nomic-embed-text).
Supports batch processing with retry logic.
"""
from __future__ import annotations

import time
from typing import List

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings


class OllamaEmbedder:
    """Generates embeddings using Ollama's embedding endpoint."""

    def __init__(self):
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.embedding_model
        self.dim = settings.embedding_dim
        self._client = httpx.Client(timeout=120.0)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        resp = self._client.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["embedding"]

    def embed_batch(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        """Embed a list of texts in batches."""
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for text in batch:
                try:
                    emb = self.embed_text(text)
                    embeddings.append(emb)
                except Exception as e:
                    logger.warning(f"Embedding failed for chunk: {e}. Using zeros.")
                    embeddings.append([0.0] * self.dim)
            logger.debug(f"Embedded {min(i + batch_size, len(texts))}/{len(texts)}")
        return embeddings

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# Singleton
_embedder: OllamaEmbedder | None = None


def get_embedder() -> OllamaEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = OllamaEmbedder()
    return _embedder
