"""
app/services/embedder.py

Thread-safe singleton wrapper around sentence-transformers.
Loaded once at application startup via the FastAPI lifespan context manager.
All embeddings are L2-normalised so that dot product == cosine similarity.

No external calls are ever made — the model weights are downloaded once on
first boot and cached in the Docker volume mounted at ~/.cache/huggingface.
"""
from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Wraps SentenceTransformer with normalised embeddings.

    Usage (singleton pattern via app lifespan):
        service = EmbeddingService(model_name="BAAI/bge-large-en-v1.5")
        vectors = service.embed(["Hello world", "Another sentence"])
        # vectors.shape == (2, 1024), dtype float32, unit-norm rows
    """

    # Class-level registry so we never load the same model twice.
    _instances: ClassVar[dict[str, "EmbeddingService"]] = {}

    def __new__(cls, model_name: str) -> "EmbeddingService":
        if model_name not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[model_name] = instance
        return cls._instances[model_name]

    def __init__(self, model_name: str) -> None:
        # Guard against re-initialisation when the singleton is retrieved.
        if hasattr(self, "_model"):
            return

        logger.info("Loading embedding model: %s (device=%s)", model_name, settings.embedding_device)
        self._model = SentenceTransformer(
            model_name,
            device=settings.embedding_device,
        )
        self._model_name = model_name
        logger.info("Embedding model loaded. Embedding dim: %d", self.embedding_dim)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def embedding_dim(self) -> int:
        return self._model.get_sentence_embedding_dimension()  # type: ignore[return-value]

    def embed(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        """
        Encode a list of strings into L2-normalised embeddings.

        Args:
            texts: Input strings.
            batch_size: Override the default batch size from settings.

        Returns:
            np.ndarray of shape (len(texts), embedding_dim), dtype float32.
            Each row is a unit vector (for cosine similarity via dot product).
        """
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        bs = batch_size or settings.embedding_batch_size
        logger.debug("Encoding %d text(s) in batches of %d", len(texts), bs)

        vectors: np.ndarray = self._model.encode(
            texts,
            batch_size=bs,
            normalize_embeddings=True,  # L2 norm → cosine sim == dot product
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)

    def embed_query(self, query: str) -> list[float]:
        """
        Convenience method for single-query embedding.
        Returns a plain Python list[float] for JSON serialisation.
        """
        vec = self.embed([query])[0]
        return vec.tolist()


# ---------------------------------------------------------------------------
# Module-level accessor — populated at app startup, used via Depends
# ---------------------------------------------------------------------------
_embedder: EmbeddingService | None = None


def get_embedder_instance() -> EmbeddingService:
    """Return the module-level singleton (must be initialised first)."""
    if _embedder is None:
        raise RuntimeError(
            "EmbeddingService has not been initialised. "
            "Ensure load_embedder() is called in the FastAPI lifespan."
        )
    return _embedder


def load_embedder() -> EmbeddingService:
    """
    Load (or retrieve) the EmbeddingService singleton.
    Called once from the FastAPI lifespan context manager.
    """
    global _embedder
    _embedder = EmbeddingService(settings.embedding_model)
    return _embedder
