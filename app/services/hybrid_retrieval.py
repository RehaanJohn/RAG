
"""
app/services/hybrid_retrieval.py

Phase 2 & Phase 4 retrieval pipeline: RRF fusion, reranking, and optional MMR diversity.

Supports both ChunkResult (text chunks) and ImageResult (images with captions).
Both implement `id`, `score`, `rerank_score`, and `rerank_content`.

Functions are pure (no I/O) and fully synchronous so they can be tested
without a running database or model. The caller (query router) is responsible
for supplying the candidate lists and model instances.

Pipeline:
    vector_results + fts_results (+ image_vector_results + image_fts_results)
        → rrf_fusion()        — merge by candidate id, compute fused RRF score
        → rerank()            — cross-encoder scoring on text/caption, sort, take top-N
        → mmr_pass()          — optional diversity selection (if MMR_ENABLED)
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Union

import numpy as np

from app.config import settings
from app.models.schemas import ChunkResult, ImageResult

if TYPE_CHECKING:
    from app.services.reranker import RerankerService

logger = logging.getLogger(__name__)

Candidate = Union[ChunkResult, ImageResult]


# ---------------------------------------------------------------------------
# 1. Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def rrf_fusion(
    *ranked_lists: list[Candidate],
    k: int | None = None,
) -> list[Candidate]:
    """
    Merge multiple ranked lists of Candidate (ChunkResult or ImageResult) using
    Reciprocal Rank Fusion.

    Score formula:  score(d) = Σ  1 / (k + rank_i(d))
    where rank_i(d) is the 1-based position of candidate d in list i.
    Candidates absent from a list contribute 0 for that list.

    Args:
        *ranked_lists: One or more lists of Candidate, each already sorted
                       by descending relevance (best first).
        k:             RRF smoothing constant (default: settings.rrf_k = 60).

    Returns:
        Deduplicated list of Candidate sorted by descending RRF score.
        The ``score`` field on each returned Candidate is the fused RRF score.
    """
    k = k if k is not None else settings.rrf_k

    # candidate_id -> Candidate (keeps last seen candidate)
    candidates_by_id: dict[str, Candidate] = {}
    # candidate_id -> accumulated RRF score
    rrf_scores: dict[str, float] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            cid = str(item.id)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in candidates_by_id:
                candidates_by_id[cid] = item

    # Build result list, attaching the fused score
    result: list[Candidate] = []
    for cid, fused_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        item = candidates_by_id[cid].model_copy(update={"score": fused_score})
        result.append(item)

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
    candidates: list[Candidate],
    reranker: "RerankerService",
    top_n: int | None = None,
) -> list[Candidate]:
    """
    Score every (query, candidate.rerank_content) pair with a CrossEncoder.
    Uses chunk content or image caption for scoring.

    Returns the top_n candidates sorted by descending rerank score.
    The ``rerank_score`` field is set on every returned Candidate.

    If candidates is empty, returns [].
    """
    top_n = top_n if top_n is not None else settings.rerank_top_n

    if not candidates:
        return []

    passages = [c.rerank_content for c in candidates]
    scores = reranker.score(query, passages)

    # Attach rerank_score to each candidate
    scored = [
        candidate.model_copy(update={"rerank_score": score})
        for candidate, score in zip(candidates, scores)
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
    candidates: list[Candidate],
    stored_embeddings: dict[str, list[float]],
    top_n: int | None = None,
    lambda_: float | None = None,
) -> list[Candidate]:
    """
    Apply Maximal Marginal Relevance over ``candidates`` to reduce near-duplicates.

    Iteratively selects the candidate maximising:
        λ · sim(candidate, query) − (1−λ) · max_j sim(candidate, selected_j)

    Args:
        query_embedding:    Query vector (same space as embeddings).
        candidates:         Pre-reranked candidates (already top-N from rerank()).
        stored_embeddings:  {id: embedding_vector} fetched from DB.
        top_n:              How many to select (default: settings.rerank_top_n).
        lambda_:            Trade-off weight (default: settings.mmr_lambda).
                            1.0 = pure relevance, 0.0 = maximum diversity.

    Returns:
        Selected candidates in MMR order (most relevant + diverse first).
        ``score`` field is updated to the MMR selection score for transparency.
    """
    top_n = top_n if top_n is not None else settings.rerank_top_n
    lambda_ = lambda_ if lambda_ is not None else settings.mmr_lambda

    if not candidates:
        return []

    q = np.array(query_embedding, dtype=np.float32)
    q_norm = q / (np.linalg.norm(q) + 1e-10)

    def _embed(item: Candidate) -> np.ndarray:
        raw = stored_embeddings.get(str(item.id), [])
        v = np.array(raw, dtype=np.float32) if raw else np.zeros_like(q)
        return v / (np.linalg.norm(v) + 1e-10)

    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    # Precompute query similarities (relevance scores)
    embeddings = {str(c.id): _embed(c) for c in candidates}
    rel_scores = {str(c.id): _cosine(embeddings[str(c.id)], q_norm) for c in candidates}

    remaining = list(candidates)
    selected: list[Candidate] = []

    while remaining and len(selected) < top_n:
        best_item: Candidate | None = None
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
                best_item = candidate.model_copy(update={"score": mmr_score})

        if best_item is None:
            break

        selected.append(best_item)
        remaining = [c for c in remaining if str(c.id) != str(best_item.id)]

    logger.debug(
        "MMR selected %d/%d candidates (λ=%.2f)", len(selected), len(candidates), lambda_
    )
    return selected
