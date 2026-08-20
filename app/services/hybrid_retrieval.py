"""
app/services/hybrid_retrieval.py

Phase 2 retrieval pipeline: RRF fusion, reranking, and optional MMR diversity.

Functions are pure (no I/O) and fully synchronous so they can be tested
without a running database or model.  The caller (query router) is responsible
for supplying the candidate lists and model instances.

Pipeline:
    vector_results + fts_results
        → rrf_fusion()        — merge by chunk id, compute fused RRF score
        → rerank()            — cross-encoder scoring, sort, take top-N
        → mmr_pass()          — optional diversity selection (if MMR_ENABLED)
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from app.config import settings
from app.models.schemas import ChunkResult

if TYPE_CHECKING:
    from app.services.reranker import RerankerService

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def rrf_fusion(
    *ranked_lists: list[ChunkResult],
    k: int | None = None,
) -> list[ChunkResult]:
    """
    Merge multiple ranked lists of ChunkResult using Reciprocal Rank Fusion.

    Score formula:  score(d) = Σ  1 / (k + rank_i(d))
    where rank_i(d) is the 1-based position of chunk d in list i.
    Chunks absent from a list contribute 0 for that list.

    Args:
        *ranked_lists: One or more lists of ChunkResult, each already sorted
                       by descending relevance (best first).
        k:             RRF smoothing constant (default: settings.rrf_k = 60).

    Returns:
        Deduplicated list of ChunkResult sorted by descending RRF score.
        The ``score`` field on each returned ChunkResult is the fused RRF score.
    """
    k = k if k is not None else settings.rrf_k

    # chunk_id -> ChunkResult (keeps last seen metadata/content)
    chunks_by_id: dict[str, ChunkResult] = {}
    # chunk_id -> accumulated RRF score
    rrf_scores: dict[str, float] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            cid = str(chunk.id)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in chunks_by_id:
                chunks_by_id[cid] = chunk

    # Build result list, attaching the fused score
    result: list[ChunkResult] = []
    for cid, fused_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        chunk = chunks_by_id[cid].model_copy(update={"score": fused_score})
        result.append(chunk)

    logger.debug(
        "RRF fusion: %d input lists, %d unique candidates",
        len(ranked_lists),
        len(result),
    )
    return result


# ---------------------------------------------------------------------------
# 2. Cross-encoder reranking
# ---------------------------------------------------------------------------


def rerank(
    query: str,
    candidates: list[ChunkResult],
    reranker: "RerankerService",
    top_n: int | None = None,
) -> list[ChunkResult]:
    """
    Score every (query, chunk.content) pair with a CrossEncoder.

    Returns the top_n chunks sorted by descending rerank score.
    The ``rerank_score`` field is set on every returned ChunkResult.

    If candidates is empty, returns [].
    """
    top_n = top_n if top_n is not None else settings.rerank_top_n

    if not candidates:
        return []

    passages = [c.content for c in candidates]
    scores = reranker.score(query, passages)

    # Attach rerank_score to each chunk
    scored = [
        chunk.model_copy(update={"rerank_score": score})
        for chunk, score in zip(candidates, scores)
    ]

    # Sort descending by rerank score
    scored.sort(key=lambda c: c.rerank_score, reverse=True)  # type: ignore[arg-type]

    logger.debug(
        "Reranked %d candidates → top-%d. Best score=%.3f, worst=%.3f",
        len(candidates),
        min(top_n, len(scored)),
        scored[0].rerank_score,
        scored[-1].rerank_score,
    )

    return scored[:top_n]


# ---------------------------------------------------------------------------
# 3. Maximal Marginal Relevance (optional diversity pass)
# ---------------------------------------------------------------------------


def mmr_pass(
    query_embedding: list[float],
    chunks: list[ChunkResult],
    stored_embeddings: dict[str, list[float]],
    top_n: int | None = None,
    lambda_: float | None = None,
) -> list[ChunkResult]:
    """
    Apply Maximal Marginal Relevance over ``chunks`` to reduce near-duplicates.

    Iteratively selects the chunk maximising:
        λ · sim(chunk, query) − (1−λ) · max_j sim(chunk, selected_j)

    Args:
        query_embedding:    Query vector (same space as chunk embeddings).
        chunks:             Pre-reranked candidates (already top-N from rerank()).
        stored_embeddings:  {chunk_id: embedding_vector} fetched from DB.
        top_n:              How many to select (default: settings.rerank_top_n).
        lambda_:            Trade-off weight (default: settings.mmr_lambda).
                            1.0 = pure relevance, 0.0 = maximum diversity.

    Returns:
        Selected chunks in MMR order (most relevant + diverse first).
        ``score`` field is updated to the MMR selection score for transparency.
    """
    top_n = top_n if top_n is not None else settings.rerank_top_n
    lambda_ = lambda_ if lambda_ is not None else settings.mmr_lambda

    if not chunks:
        return []

    q = np.array(query_embedding, dtype=np.float32)
    q_norm = q / (np.linalg.norm(q) + 1e-10)

    def _embed(chunk: ChunkResult) -> np.ndarray:
        raw = stored_embeddings.get(str(chunk.id), [])
        v = np.array(raw, dtype=np.float32) if raw else np.zeros_like(q)
        return v / (np.linalg.norm(v) + 1e-10)

    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    # Precompute query similarities (relevance scores)
    embeddings = {str(c.id): _embed(c) for c in chunks}
    rel_scores = {str(c.id): _cosine(embeddings[str(c.id)], q_norm) for c in chunks}

    remaining = list(chunks)
    selected: list[ChunkResult] = []

    while remaining and len(selected) < top_n:
        best_chunk: ChunkResult | None = None
        best_score = float("-inf")

        for candidate in remaining:
            cid = str(candidate.id)
            relevance = rel_scores[cid]

            if not selected:
                redundancy = 0.0
            else:
                redundancy = max(
                    _cosine(embeddings[cid], embeddings[str(s.id)])
                    for s in selected
                )

            mmr_score = lambda_ * relevance - (1.0 - lambda_) * redundancy

            if mmr_score > best_score:
                best_score = mmr_score
                best_chunk = candidate.model_copy(update={"score": mmr_score})

        if best_chunk is None:
            break

        selected.append(best_chunk)
        remaining = [c for c in remaining if str(c.id) != str(best_chunk.id)]

    logger.debug(
        "MMR selected %d/%d chunks (λ=%.2f)", len(selected), len(chunks), lambda_
    )
    return selected
