"""
app/services/reranker.py

Local cross-encoder reranker singleton.

Uses sentence_transformers.CrossEncoder (BAAI/bge-reranker-large by default).
Loaded once at startup in main.py lifespan; injected into requests via
app/dependencies.py.

CrossEncoder scores are raw logits (unbounded, not normalised to [0,1]).
Higher score = more relevant.  The gate threshold (GATE_RERANK_THRESHOLD)
must be calibrated empirically for the specific corpus and model being used.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.config import settings

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder as _CrossEncoderType

logger = logging.getLogger(__name__)

# Module-level singleton — set by load_reranker(), read by get_reranker_instance()
_reranker_instance: "RerankerService | None" = None


class RerankerService:
    """
    Thin wrapper around sentence_transformers.CrossEncoder.

    Thread-safety: CrossEncoder.predict() holds the GIL for the forward pass
    but is otherwise safe to call from multiple asyncio tasks via
    run_in_executor since it is CPU-bound.
    """

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        logger.info("Loading cross-encoder reranker: %s", model_name)
        self._model: _CrossEncoderType = CrossEncoder(
            model_name,
            max_length=512,       # Truncate long passage+query pairs
            device=settings.embedding_device,
        )
        self._model_name = model_name
        logger.info("Reranker loaded: %s", model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    def score(self, query: str, passages: list[str]) -> list[float]:
        """
        Score each (query, passage) pair.

        Returns a list of floats parallel to `passages`.
        Scores are raw CrossEncoder logits — higher = more relevant.
        NOT normalised to [0,1].
        """
        if not passages:
            return []
        pairs = [(query, p) for p in passages]
        scores = self._model.predict(pairs)
        # predict() returns numpy array or list depending on version
        return [float(s) for s in scores]


def load_reranker() -> RerankerService:
    """
    Load (or return cached) the reranker singleton.
    Called once from main.py lifespan.
    """
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = RerankerService(settings.rerank_model)
    return _reranker_instance


def get_reranker_instance() -> "RerankerService":
    """Return the already-loaded singleton. Raises if not yet initialised."""
    if _reranker_instance is None:
        raise RuntimeError(
            "RerankerService not initialised. "
            "Ensure load_reranker() is called in the lifespan before requests."
        )
    return _reranker_instance
