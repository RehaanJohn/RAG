"""
app/models/db.py
Raw SQL queries and asyncpg helpers for all DB operations.
We use asyncpg directly (no ORM) for minimal overhead on vector queries.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

import asyncpg

from app.models.schemas import ChunkResult, DocStatus, DocumentResponse


# ---------------------------------------------------------------------------
# documents table
# ---------------------------------------------------------------------------


async def create_document(
    conn: asyncpg.Connection,
    *,
    filename: str,
    source_type: str,
    tenant_id: uuid.UUID,
    metadata: dict[str, Any] | None = None,
) -> uuid.UUID:
    """Insert a new document row in 'pending' status. Returns its UUID."""
    row = await conn.fetchrow(
        """
        INSERT INTO documents (filename, source_type, tenant_id, metadata)
        VALUES ($1, $2, $3, $4)
        RETURNING id
        """,
        filename,
        source_type,
        tenant_id,
        metadata or {},
    )
    return row["id"]


async def get_document(
    conn: asyncpg.Connection,
    doc_id: uuid.UUID,
) -> DocumentResponse | None:
    row = await conn.fetchrow(
        """
        SELECT id, filename, source_type, status, tenant_id,
               created_at, error_message, metadata
        FROM documents
        WHERE id = $1
        """,
        doc_id,
    )
    if row is None:
        return None
    return DocumentResponse(
        id=row["id"],
        filename=row["filename"],
        source_type=row["source_type"],
        status=DocStatus(row["status"]),
        tenant_id=row["tenant_id"],
        created_at=row["created_at"],
        error_message=row["error_message"],
        metadata=dict(row["metadata"]),
    )


async def update_document_status(
    conn: asyncpg.Connection,
    doc_id: uuid.UUID,
    status: DocStatus,
    error_message: str | None = None,
) -> None:
    await conn.execute(
        """
        UPDATE documents
        SET status = $1, error_message = $2
        WHERE id = $3
        """,
        status.value,
        error_message,
        doc_id,
    )


# ---------------------------------------------------------------------------
# chunks table
# ---------------------------------------------------------------------------


async def upsert_chunk(
    conn: asyncpg.Connection,
    *,
    doc_id: uuid.UUID,
    content: str,
    content_hash: str,
    embedding: list[float],
    metadata: dict[str, Any],
) -> None:
    """
    Insert a chunk. If a chunk with the same content_hash already exists
    (deduplication), the row is silently skipped (ON CONFLICT DO NOTHING).
    """
    await conn.execute(
        """
        INSERT INTO chunks (doc_id, content, content_hash, embedding, metadata)
        VALUES ($1, $2, $3, $4::vector, $5)
        ON CONFLICT (content_hash) DO NOTHING
        """,
        doc_id,
        content,
        content_hash,
        str(embedding),
        metadata,
    )


async def similarity_search(
    conn: asyncpg.Connection,
    *,
    query_embedding: list[float],
    tenant_id: uuid.UUID,
    top_k: int,
) -> list[ChunkResult]:
    """
    Cosine similarity search via pgvector.
    Returns chunks ordered by descending similarity, filtered to tenant_id.
    Score = 1 − cosine_distance.
    """
    rows = await conn.fetch(
        """
        SELECT
            c.id,
            c.doc_id,
            c.content,
            c.metadata,
            1 - (c.embedding <=> $1::vector) AS score
        FROM chunks c
        JOIN documents d ON d.id = c.doc_id
        WHERE d.tenant_id = $2
        ORDER BY c.embedding <=> $1::vector
        LIMIT $3
        """,
        str(query_embedding),
        tenant_id,
        top_k,
    )
    return [
        ChunkResult(
            id=row["id"],
            doc_id=row["doc_id"],
            content=row["content"],
            metadata=dict(row["metadata"]),
            score=float(row["score"]),
        )
        for row in rows
    ]


async def full_text_search(
    conn: asyncpg.Connection,
    *,
    query: str,
    tenant_id: uuid.UUID,
    top_k: int,
) -> list[ChunkResult]:
    """
    Full-text keyword search via Postgres tsvector + ts_rank_cd.

    Uses websearch_to_tsquery for natural boolean syntax support
    (AND/OR/phrase/negation) without SQL injection risk.
    Chunks are filtered by tenant_id via a JOIN to documents.

    Returns an empty list if the query produces no tsquery tokens
    (e.g. single stop-word queries).
    """
    try:
        rows = await conn.fetch(
            """
            SELECT
                c.id,
                c.doc_id,
                c.content,
                c.metadata,
                ts_rank_cd(c.content_tsv,
                           websearch_to_tsquery('english', $1)) AS score
            FROM chunks c
            JOIN documents d ON d.id = c.doc_id
            WHERE d.tenant_id = $2
              AND c.content_tsv @@ websearch_to_tsquery('english', $1)
            ORDER BY score DESC
            LIMIT $3
            """,
            query,
            tenant_id,
            top_k,
        )
    except asyncpg.PostgresError:
        # e.g. syntax error in query; degrade gracefully to empty results
        return []

    return [
        ChunkResult(
            id=row["id"],
            doc_id=row["doc_id"],
            content=row["content"],
            metadata=dict(row["metadata"]),
            score=float(row["score"]),
        )
        for row in rows
    ]


async def get_chunk_embeddings(
    conn: asyncpg.Connection,
    chunk_ids: list[uuid.UUID],
) -> dict[str, list[float]]:
    """
    Fetch stored embeddings for a list of chunk IDs.
    Used by the MMR diversity pass which needs inter-chunk cosine similarities.
    Returns {str(chunk_id): embedding_vector}.
    """
    if not chunk_ids:
        return {}
    rows = await conn.fetch(
        """
        SELECT id, embedding::text AS embedding_text
        FROM chunks
        WHERE id = ANY($1::uuid[])
        """,
        chunk_ids,
    )
    result: dict[str, list[float]] = {}
    for row in rows:
        # asyncpg returns the vector as a text string like "[0.1,0.2,...]"
        emb_text: str = row["embedding_text"]
        floats = [float(x) for x in emb_text.strip("[]").split(",")]
        result[str(row["id"])] = floats
    return result


async def get_document_map(
    conn: asyncpg.Connection,
    doc_ids: list[uuid.UUID],
) -> dict[uuid.UUID, DocumentResponse]:
    """Fetch minimal document info for citation resolution."""
    if not doc_ids:
        return {}
    rows = await conn.fetch(
        """
        SELECT id, filename, source_type, status, tenant_id, created_at, metadata
        FROM documents
        WHERE id = ANY($1::uuid[])
        """,
        doc_ids,
    )
    result: dict[uuid.UUID, DocumentResponse] = {}
    for row in rows:
        result[row["id"]] = DocumentResponse(
            id=row["id"],
            filename=row["filename"],
            source_type=row["source_type"],
            status=DocStatus(row["status"]),
            tenant_id=row["tenant_id"],
            created_at=row["created_at"],
            metadata=dict(row["metadata"]),
        )
    return result


# ---------------------------------------------------------------------------
# query_logs table
# ---------------------------------------------------------------------------


async def log_query(
    conn: asyncpg.Connection,
    *,
    query: str,
    tenant_id: uuid.UUID | None,
    retrieved_chunk_ids: list[str],
    top_similarity_score: float | None,
    answer: str | None,
    was_refused: bool,
    latency_ms: int,
    # Phase 2 additions (optional — default None for backwards compat)
    rerank_scores: dict[str, float] | None = None,
    retrieval_candidate_count: int | None = None,
) -> None:
    await conn.execute(
        """
        INSERT INTO query_logs
            (query, tenant_id, retrieved_chunk_ids, top_similarity_score,
             answer, was_refused, latency_ms,
             rerank_scores, retrieval_candidate_count)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """,
        query,
        tenant_id,
        retrieved_chunk_ids,
        top_similarity_score,
        answer,
        was_refused,
        latency_ms,
        rerank_scores,        # JSONB codec handles dict → jsonb
        retrieval_candidate_count,
    )
