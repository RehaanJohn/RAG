"""
app/worker/tasks.py

arq background job: ingest_document

Pipeline stages:
  parsing → chunking → embedding → indexing

Each stage updates documents.status so the API can report progress.
On any unhandled exception the document is marked 'failed' with the error message.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

import asyncpg

from app.config import settings
from app.models.db import (
    get_document,
    update_document_status,
    upsert_chunk,
)
from app.models.schemas import DocStatus
from app.services.embedder import EmbeddingService, load_embedder
from app.worker.chunker import StructureAwareChunker
from app.worker.parsers import parse_file

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# arq job
# ---------------------------------------------------------------------------

async def ingest_document(ctx: dict, doc_id: str) -> None:
    """
    arq background job: parse → chunk → embed → index a single document.

    ctx is populated by arq from WorkerSettings.ctx_factory.
    Expected ctx keys:
      - "db_pool": asyncpg.Pool
      - "embedder": EmbeddingService
      - "chunker": StructureAwareChunker
    """
    doc_uuid = uuid.UUID(doc_id)
    logger.info("Starting ingestion for doc_id=%s", doc_id)

    # Acquire a DB connection with service-role context (bypasses RLS)
    pool: asyncpg.Pool = ctx["db_pool"]

    async with pool.acquire() as conn:
        # Service-role bypass so the worker can read/write all tenants
        await conn.execute("SELECT set_service_role_context()")

        # ---------------------------------------------------------------
        # 1. Fetch document row
        # ---------------------------------------------------------------
        doc = await get_document(conn, doc_uuid)
        if doc is None:
            logger.error("Document %s not found — aborting job.", doc_id)
            return

        raw_path = Path(settings.raw_files_dir) / str(doc.tenant_id) / f"{doc_id}_{doc.filename}"
        if not raw_path.exists():
            await update_document_status(
                conn, doc_uuid, DocStatus.failed,
                error_message=f"Raw file not found: {raw_path}",
            )
            return

        try:
            # -----------------------------------------------------------
            # 2. Parse
            # -----------------------------------------------------------
            await update_document_status(conn, doc_uuid, DocStatus.parsing)
            parsed = parse_file(raw_path, doc.source_type)
            logger.info("Parsed %d page(s) from %s", len(parsed.pages), doc.filename)

            # -----------------------------------------------------------
            # 3. Chunk
            # -----------------------------------------------------------
            await update_document_status(conn, doc_uuid, DocStatus.chunking)
            chunker: StructureAwareChunker = ctx["chunker"]
            chunks = chunker.chunk_document(parsed)
            logger.info("Produced %d chunk(s) for doc %s", len(chunks), doc_id)

            if not chunks:
                await update_document_status(
                    conn, doc_uuid, DocStatus.failed,
                    error_message="No chunks produced — document may be empty or unreadable.",
                )
                return

            # -----------------------------------------------------------
            # 4. Embed
            # -----------------------------------------------------------
            await update_document_status(conn, doc_uuid, DocStatus.embedding)
            embedder: EmbeddingService = ctx["embedder"]
            texts = [c.content for c in chunks]
            vectors = embedder.embed(texts, batch_size=settings.embedding_batch_size)
            logger.info("Embedded %d chunk(s).", len(chunks))

            # -----------------------------------------------------------
            # 5. Index (upsert with dedup)
            # -----------------------------------------------------------
            inserted = 0
            skipped = 0
            for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
                meta = {
                    **chunk.metadata,
                    "chunk_index": chunk.metadata.get("chunk_index", i),
                }
                try:
                    await upsert_chunk(
                        conn,
                        doc_id=doc_uuid,
                        content=chunk.content,
                        content_hash=chunk.content_hash,
                        embedding=vector.tolist(),
                        metadata=meta,
                    )
                    inserted += 1
                except asyncpg.UniqueViolationError:
                    # Duplicate content — already indexed from another doc
                    skipped += 1
                    logger.debug("Skipped duplicate chunk (hash=%s)", chunk.content_hash)

            logger.info(
                "Indexed %d chunk(s), skipped %d duplicate(s) for doc %s.",
                inserted, skipped, doc_id,
            )

            # -----------------------------------------------------------
            # 6. Mark indexed
            # -----------------------------------------------------------
            await update_document_status(conn, doc_uuid, DocStatus.indexed)
            logger.info("Document %s is now indexed.", doc_id)

        except Exception as exc:
            logger.exception("Ingestion failed for doc %s: %s", doc_id, exc)
            await update_document_status(
                conn, doc_uuid, DocStatus.failed,
                error_message=str(exc)[:1024],
            )
            raise  # Let arq record the failure
