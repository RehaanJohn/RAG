"""
app/routers/query.py

POST /query — Phase 2 pipeline:
  1.  Embed query locally.
  2a. Vector cosine similarity search  ─┐
  2b. Full-text (tsvector) search       ├─ run in parallel, each returns top-40
  3.  RRF fusion → deduplicated candidate list.
  4.  Cross-encoder reranking → sorted top-N.
  5.  Confidence gate: top rerank score < GATE_RERANK_THRESHOLD → refuse.
  6.  Optional MMR diversity pass (MMR_ENABLED env var).
  7.  Assemble context, call local Ollama, validate citations.
  8.  Log to query_logs (with rerank_scores + candidate_count).
  9.  Return answer + sources + Phase 2 metadata.

API contract is unchanged from Phase 1.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, status

from app.config import settings
from app.dependencies import get_db_pool, get_embedder, get_reranker
from app.models.db import (
    full_text_search,
    get_chunk_embeddings,
    get_document_map,
    log_query,
    similarity_search,
)
from app.models.schemas import ChunkResult, QueryRequest, QueryResponse, SourceObject
from app.services.citation import resolve_citations
from app.services.embedder import EmbeddingService
from app.services.generation import REFUSAL_MESSAGE, generate_answer
from app.services.hybrid_retrieval import mmr_pass, rerank, rrf_fusion
from app.services.reranker import RerankerService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/query", tags=["query"])


@router.post(
    "",
    response_model=QueryResponse,
    summary="Ask a question against the indexed knowledge base.",
)
async def query_endpoint(
    request: QueryRequest,
    pool: asyncpg.Pool = Depends(get_db_pool),
    embedder: EmbeddingService = Depends(get_embedder),
    reranker: RerankerService = Depends(get_reranker),
) -> QueryResponse:
    """
    Phase 2 RAG pipeline with hybrid retrieval and cross-encoder reranking.
    """
    t_start = time.monotonic()
    candidate_k = settings.hybrid_candidate_k

    # -----------------------------------------------------------------------
    # 1. Embed query (local, in-process)
    # -----------------------------------------------------------------------
    query_vector = embedder.embed_query(request.query)

    # -----------------------------------------------------------------------
    # 2. Parallel retrieval: vector search + full-text search
    #    Each uses its own DB connection to avoid asyncpg single-connection
    #    concurrency restrictions.
    # -----------------------------------------------------------------------
    async def _vector_search(conn: asyncpg.Connection) -> list[ChunkResult]:
        await conn.execute("SELECT set_tenant_context($1::uuid)", request.tenant_id)
        return await similarity_search(
            conn,
            query_embedding=query_vector,
            tenant_id=request.tenant_id,
            top_k=candidate_k,
        )

    async def _fts_search(conn: asyncpg.Connection) -> list[ChunkResult]:
        await conn.execute("SELECT set_tenant_context($1::uuid)", request.tenant_id)
        return await full_text_search(
            conn,
            query=request.query,
            tenant_id=request.tenant_id,
            top_k=candidate_k,
        )

    async with pool.acquire() as conn_vec, pool.acquire() as conn_fts:
        vector_results, fts_results = await asyncio.gather(
            _vector_search(conn_vec),
            _fts_search(conn_fts),
        )

    logger.info(
        "Query='%.80s', tenant=%s | vector_hits=%d, fts_hits=%d",
        request.query, request.tenant_id,
        len(vector_results), len(fts_results),
    )

    # -----------------------------------------------------------------------
    # 3. RRF fusion — merge and deduplicate by chunk id
    # -----------------------------------------------------------------------
    fused_candidates = rrf_fusion(vector_results, fts_results, k=settings.rrf_k)
    candidate_count = len(fused_candidates)

    if not fused_candidates:
        # Nothing retrieved at all — refuse immediately
        latency_ms = int((time.monotonic() - t_start) * 1000)
        async with pool.acquire() as log_conn:
            await log_conn.execute(
                "SELECT set_tenant_context($1::uuid)", request.tenant_id
            )
            await log_query(
                log_conn,
                query=request.query,
                tenant_id=request.tenant_id,
                retrieved_chunk_ids=[],
                top_similarity_score=None,
                answer=REFUSAL_MESSAGE,
                was_refused=True,
                latency_ms=latency_ms,
                rerank_scores=None,
                retrieval_candidate_count=0,
            )
        return QueryResponse(
            query=request.query,
            answer=REFUSAL_MESSAGE,
            was_refused=True,
            top_similarity_score=None,
            sources=[],
            latency_ms=latency_ms,
            rerank_scores=None,
            retrieval_candidate_count=0,
        )

    # -----------------------------------------------------------------------
    # 4. Cross-encoder reranking (CPU-bound; runs synchronously in thread pool
    #    via FastAPI's default executor — acceptable for moderate candidate sets)
    # -----------------------------------------------------------------------
    top_chunks = rerank(
        request.query,
        fused_candidates,
        reranker,
        top_n=settings.rerank_top_n,
    )

    top_rerank_score: float = top_chunks[0].rerank_score or float("-inf")
    rerank_score_map: dict[str, float] = {
        str(c.id): c.rerank_score  # type: ignore[misc]
        for c in top_chunks
        if c.rerank_score is not None
    }

    logger.info(
        "Reranked %d→%d. top_rerank_score=%.3f, gate_threshold=%.3f",
        candidate_count, len(top_chunks),
        top_rerank_score, settings.gate_rerank_threshold,
    )

    # -----------------------------------------------------------------------
    # 5. Confidence gate — now based on reranker score, not cosine similarity
    # -----------------------------------------------------------------------
    if top_rerank_score < settings.gate_rerank_threshold:
        latency_ms = int((time.monotonic() - t_start) * 1000)
        logger.info(
            "Query refused. top_rerank_score=%.3f < gate_threshold=%.3f. latency=%dms",
            top_rerank_score, settings.gate_rerank_threshold, latency_ms,
        )
        async with pool.acquire() as log_conn:
            await log_conn.execute(
                "SELECT set_tenant_context($1::uuid)", request.tenant_id
            )
            await log_query(
                log_conn,
                query=request.query,
                tenant_id=request.tenant_id,
                retrieved_chunk_ids=[str(c.id) for c in fused_candidates],
                top_similarity_score=fused_candidates[0].score if fused_candidates else None,
                answer=REFUSAL_MESSAGE,
                was_refused=True,
                latency_ms=latency_ms,
                rerank_scores=rerank_score_map,
                retrieval_candidate_count=candidate_count,
            )
        return QueryResponse(
            query=request.query,
            answer=REFUSAL_MESSAGE,
            was_refused=True,
            top_similarity_score=fused_candidates[0].score if fused_candidates else None,
            sources=[],
            latency_ms=latency_ms,
            rerank_scores=rerank_score_map,
            retrieval_candidate_count=candidate_count,
        )

    # -----------------------------------------------------------------------
    # 6. Optional MMR diversity pass
    # -----------------------------------------------------------------------
    final_chunks: list[ChunkResult]
    if settings.mmr_enabled and len(top_chunks) > 1:
        chunk_ids = [c.id for c in top_chunks]
        async with pool.acquire() as emb_conn:
            await emb_conn.execute(
                "SELECT set_tenant_context($1::uuid)", request.tenant_id
            )
            stored_embeddings = await get_chunk_embeddings(emb_conn, chunk_ids)

        final_chunks = mmr_pass(
            query_embedding=query_vector,
            chunks=top_chunks,
            stored_embeddings=stored_embeddings,
            top_n=settings.rerank_top_n,
            lambda_=settings.mmr_lambda,
        )
        logger.info("MMR applied: %d → %d chunks", len(top_chunks), len(final_chunks))
    else:
        final_chunks = top_chunks

    # -----------------------------------------------------------------------
    # 7. Fetch document metadata for citation resolution
    # -----------------------------------------------------------------------
    doc_ids = list({c.doc_id for c in final_chunks})
    async with pool.acquire() as doc_conn:
        await doc_conn.execute(
            "SELECT set_tenant_context($1::uuid)", request.tenant_id
        )
        documents = await get_document_map(doc_conn, doc_ids)

    # -----------------------------------------------------------------------
    # 8. Generate answer (local Ollama) with citation validation
    # -----------------------------------------------------------------------
    answer, sources, was_refused = await generate_answer(
        query=request.query,
        chunks=final_chunks,
        documents=documents,
    )

    latency_ms = int((time.monotonic() - t_start) * 1000)

    # -----------------------------------------------------------------------
    # 9. Log query with Phase 2 metadata
    # -----------------------------------------------------------------------
    async with pool.acquire() as log_conn:
        await log_conn.execute(
            "SELECT set_tenant_context($1::uuid)", request.tenant_id
        )
        await log_query(
            log_conn,
            query=request.query,
            tenant_id=request.tenant_id,
            retrieved_chunk_ids=[str(c.id) for c in final_chunks],
            top_similarity_score=fused_candidates[0].score if fused_candidates else None,
            answer=answer,
            was_refused=was_refused,
            latency_ms=latency_ms,
            rerank_scores=rerank_score_map,
            retrieval_candidate_count=candidate_count,
        )

    logger.info(
        "Query complete. was_refused=%s, sources=%d, candidates=%d, latency=%dms",
        was_refused, len(sources), candidate_count, latency_ms,
    )

    return QueryResponse(
        query=request.query,
        answer=answer,
        was_refused=was_refused,
        top_similarity_score=fused_candidates[0].score if fused_candidates else None,
        sources=sources,
        latency_ms=latency_ms,
        rerank_scores=rerank_score_map,
        retrieval_candidate_count=candidate_count,
    )
