"""
app/routers/query.py

POST /query — Phase 2 & Phase 4 pipeline:
  0.  [Phase 4] If conversation_id provided + recent images in Redis:
        → Classify if query is asking about a previously-shown image.
        → If yes: route to direct VLM visual Q&A.
  1.  Embed query locally.
  2.  Parallel retrieval across connections:
        - Vector cosine similarity search on chunks
        - Full-text (tsvector) search on chunks
        - [Phase 4] Vector cosine similarity search on images
        - [Phase 4] Full-text (tsvector) search on image captions
  3.  RRF fusion across all result lists → deduplicated candidate list.
  4.  Cross-encoder reranking on content/captions → sorted top-N.
  5.  Confidence gate: top rerank score < GATE_RERANK_THRESHOLD → refuse.
  6.  Optional MMR diversity pass (MMR_ENABLED env var).
  7.  Assemble multimodal context (text + image captions), call local Ollama, validate citations.
  8.  Log to query_logs (with rerank_scores + candidate_count).
  9.  [Phase 4] Track surfaced images in Redis for conversation session.
  10. Return answer + sources (text and inline images) + metadata.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.config import settings
from app.dependencies import get_db_pool, get_embedder, get_reranker
from app.models.db import (
    full_text_search,
    full_text_search_images,
    get_chunk_embeddings,
    get_document,
    get_document_map,
    get_image,
    log_query,
    similarity_search,
    similarity_search_images,
)
from app.models.schemas import (
    ChunkResult,
    ImageResult,
    QueryRequest,
    QueryResponse,
    SourceObject,
)
from app.services.citation import resolve_citations
from app.services.embedder import EmbeddingService
from app.services.generation import REFUSAL_MESSAGE, generate_answer
from app.services.hybrid_retrieval import Candidate, mmr_pass, rerank, rrf_fusion
from app.services.image_store import load_image, storage_path_to_url
from app.services.reranker import RerankerService
from app.services.vlm import ask_image, classify_image_reference

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/query", tags=["query"])


@router.post(
    "",
    response_model=QueryResponse,
    summary="Ask a question against the indexed knowledge base.",
)
async def query_endpoint(
    request: QueryRequest,
    raw_request: Request,
    pool: asyncpg.Pool = Depends(get_db_pool),
    embedder: EmbeddingService = Depends(get_embedder),
    reranker: RerankerService = Depends(get_reranker),
) -> QueryResponse:
    """
    Phase 4 Multimodal RAG pipeline with hybrid retrieval, cross-encoder reranking,
    and conversational visual Q&A auto-routing.
    """
    t_start = time.monotonic()
    candidate_k = settings.hybrid_candidate_k

    # -----------------------------------------------------------------------
    # 0. Conversation image follow-up auto-routing (Phase 4)
    # -----------------------------------------------------------------------
    if request.conversation_id and settings.image_ref_classify_enabled:
        try:
            redis = raw_request.app.state.redis
            history_key = f"img_history:{request.conversation_id}"
            recent_raw_ids = await redis.lrange(history_key, 0, 4)
            if recent_raw_ids:
                # Fetch recent image captions
                recent_images: list[ImageResult] = []
                async with pool.acquire() as conn:
                    await conn.execute("SELECT set_tenant_context($1::uuid)", request.tenant_id)
                    for raw_id in recent_raw_ids:
                        img_uuid = uuid.UUID(raw_id.decode("utf-8") if isinstance(raw_id, bytes) else str(raw_id))
                        img = await get_image(conn, img_uuid, request.tenant_id)
                        if img:
                            recent_images.append(img)

                if recent_images:
                    captions = [img.caption or "Image" for img in recent_images]
                    is_ref, match_idx = await classify_image_reference(request.query, captions)
                    if is_ref and match_idx is not None and match_idx < len(recent_images):
                        target_img = recent_images[match_idx]
                        logger.info(
                            "Routing query to direct visual Q&A on image %s", target_img.id
                        )
                        img_bytes = load_image(target_img.storage_path)
                        answer, vlm_unavail = await ask_image(img_bytes, request.query)
                        latency_ms = int((time.monotonic() - t_start) * 1000)

                        if not vlm_unavail and answer:
                            doc_map = {}
                            async with pool.acquire() as conn:
                                await conn.execute("SELECT set_tenant_context($1::uuid)", request.tenant_id)
                                doc = await get_document(conn, target_img.doc_id)
                                filename = doc.filename if doc else "unknown"

                            source = SourceObject(
                                type="image",
                                citation_id=target_img.citation_id,
                                filename=filename,
                                page_number=target_img.page_number,
                                score=1.0,
                                image_url=storage_path_to_url(target_img.storage_path),
                            )

                            return QueryResponse(
                                query=request.query,
                                answer=answer,
                                was_refused=False,
                                top_similarity_score=1.0,
                                sources=[source],
                                latency_ms=latency_ms,
                                rerank_scores={str(target_img.id): 1.0},
                                retrieval_candidate_count=1,
                                routed_to_image_qa=True,
                            )
        except Exception as exc:
            logger.warning("Conversation image check failed (falling back to standard RAG): %s", exc)

    # -----------------------------------------------------------------------
    # 1. Embed query (local, in-process)
    # -----------------------------------------------------------------------
    query_vector = embedder.embed_query(request.query)

    # -----------------------------------------------------------------------
    # 2. Parallel retrieval: vector + full-text for text chunks AND images
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

    async def _image_vector_search(conn: asyncpg.Connection) -> list[ImageResult]:
        if not settings.image_search_enabled:
            return []
        await conn.execute("SELECT set_tenant_context($1::uuid)", request.tenant_id)
        return await similarity_search_images(
            conn,
            query_embedding=query_vector,
            tenant_id=request.tenant_id,
            top_k=candidate_k,
        )

    async def _image_fts_search(conn: asyncpg.Connection) -> list[ImageResult]:
        if not settings.image_search_enabled:
            return []
        await conn.execute("SELECT set_tenant_context($1::uuid)", request.tenant_id)
        return await full_text_search_images(
            conn,
            query=request.query,
            tenant_id=request.tenant_id,
            top_k=candidate_k,
        )

    async with pool.acquire() as conn_vec, \
               pool.acquire() as conn_fts, \
               pool.acquire() as conn_img_vec, \
               pool.acquire() as conn_img_fts:
        vector_results, fts_results, img_vec_results, img_fts_results = await asyncio.gather(
            _vector_search(conn_vec),
            _fts_search(conn_fts),
            _image_vector_search(conn_img_vec),
            _image_fts_search(conn_img_fts),
        )

    logger.info(
        "Query='%.80s', tenant=%s | vector_chunks=%d, fts_chunks=%d, vector_images=%d, fts_images=%d",
        request.query, request.tenant_id,
        len(vector_results), len(fts_results), len(img_vec_results), len(img_fts_results),
    )

    # -----------------------------------------------------------------------
    # 3. RRF fusion — merge and deduplicate across all 4 result lists
    # -----------------------------------------------------------------------
    fused_candidates: list[Candidate] = rrf_fusion(
        vector_results,
        fts_results,
        img_vec_results,
        img_fts_results,
        k=settings.rrf_k,
    )
    candidate_count = len(fused_candidates)

    if not fused_candidates:
        latency_ms = int((time.monotonic() - t_start) * 1000)
        async with pool.acquire() as log_conn:
            await log_conn.execute("SELECT set_tenant_context($1::uuid)", request.tenant_id)
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
    # 4. Cross-encoder reranking
    # -----------------------------------------------------------------------
    top_candidates = rerank(
        request.query,
        fused_candidates,
        reranker,
        top_n=settings.rerank_top_n,
    )

    top_rerank_score: float = top_candidates[0].rerank_score or float("-inf")
    rerank_score_map: dict[str, float] = {
        str(c.id): c.rerank_score  # type: ignore[misc]
        for c in top_candidates
        if c.rerank_score is not None
    }

    logger.info(
        "Reranked %d→%d. top_rerank_score=%.3f, gate_threshold=%.3f",
        candidate_count, len(top_candidates),
        top_rerank_score, settings.gate_rerank_threshold,
    )

    # -----------------------------------------------------------------------
    # 5. Confidence gate
    # -----------------------------------------------------------------------
    if top_rerank_score < settings.gate_rerank_threshold:
        latency_ms = int((time.monotonic() - t_start) * 1000)
        logger.info(
            "Query refused. top_rerank_score=%.3f < gate_threshold=%.3f. latency=%dms",
            top_rerank_score, settings.gate_rerank_threshold, latency_ms,
        )
        async with pool.acquire() as log_conn:
            await log_conn.execute("SELECT set_tenant_context($1::uuid)", request.tenant_id)
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
    final_candidates: list[Candidate]
    if settings.mmr_enabled and len(top_candidates) > 1:
        chunk_ids = [c.id for c in top_candidates if isinstance(c, ChunkResult)]
        async with pool.acquire() as emb_conn:
            await emb_conn.execute("SELECT set_tenant_context($1::uuid)", request.tenant_id)
            stored_embeddings = await get_chunk_embeddings(emb_conn, chunk_ids)

        final_candidates = mmr_pass(
            query_embedding=query_vector,
            candidates=top_candidates,
            stored_embeddings=stored_embeddings,
            top_n=settings.rerank_top_n,
            lambda_=settings.mmr_lambda,
        )
        logger.info("MMR applied: %d → %d candidates", len(top_candidates), len(final_candidates))
    else:
        final_candidates = top_candidates

    # Split final candidates into chunks and images
    final_chunks: list[ChunkResult] = [c for c in final_candidates if isinstance(c, ChunkResult)]
    final_images: list[ImageResult] = [c for c in final_candidates if isinstance(c, ImageResult)]

    # -----------------------------------------------------------------------
    # 7. Fetch document metadata for citation resolution
    # -----------------------------------------------------------------------
    doc_ids = list({c.doc_id for c in final_candidates})
    async with pool.acquire() as doc_conn:
        await doc_conn.execute("SELECT set_tenant_context($1::uuid)", request.tenant_id)
        documents = await get_document_map(doc_conn, doc_ids)

    # -----------------------------------------------------------------------
    # 8. Generate answer (local Ollama) with citation validation
    # -----------------------------------------------------------------------
    answer, sources, was_refused = await generate_answer(
        query=request.query,
        chunks=final_chunks,
        documents=documents,
        images=final_images,
    )

    latency_ms = int((time.monotonic() - t_start) * 1000)

    # -----------------------------------------------------------------------
    # 9. Track surfaced images in Redis for conversation history
    # -----------------------------------------------------------------------
    if request.conversation_id:
        surfaced_images = [s for s in sources if s.type == "image"]
        if surfaced_images:
            try:
                redis = raw_request.app.state.redis
                key = f"img_history:{request.conversation_id}"
                for s in surfaced_images:
                    # Find corresponding image_id
                    for img in final_images:
                        if img.citation_id == s.citation_id:
                            await redis.lpush(key, str(img.id))
                await redis.ltrim(key, 0, 4)
                await redis.expire(key, 3600)
            except Exception as exc:
                logger.warning("Failed to store surfaced images to Redis: %s", exc)

    # -----------------------------------------------------------------------
    # 10. Log query with Phase 4 metadata
    # -----------------------------------------------------------------------
    async with pool.acquire() as log_conn:
        await log_conn.execute("SELECT set_tenant_context($1::uuid)", request.tenant_id)
        await log_query(
            log_conn,
            query=request.query,
            tenant_id=request.tenant_id,
            retrieved_chunk_ids=[str(c.id) for c in final_candidates],
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
