"""
app/worker/tasks.py

arq background job: ingest_document

Pipeline stages:
  parsing → chunking → embedding → indexing → image extraction (PDF only)

Each stage updates documents.status so the API can report progress.
On any unhandled exception the document is marked 'failed' with the error message.

Phase 4: Image extraction runs AFTER text indexing. If it fails for individual
images, those images are skipped — the document is still marked 'indexed'.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

import asyncpg

from app.config import settings
from app.models.db import (
    find_nearby_chunk_id,
    get_document,
    insert_image,
    update_document_status,
    update_image_caption,
    upsert_chunk,
)
from app.models.schemas import DocStatus
from app.services.embedder import EmbeddingService, load_embedder
from app.services.image_store import save_image
from app.services.vlm import caption_image
from app.worker.chunker import StructureAwareChunker
from app.worker.image_extractor import extract_images_from_pdf
from app.worker.parsers import parse_file

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# arq job
# ---------------------------------------------------------------------------

async def ingest_document(ctx: dict, doc_id: str) -> None:
    """
    arq background job: parse → chunk → embed → index → extract images.

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
                    skipped += 1
                    logger.debug("Skipped duplicate chunk (hash=%s)", chunk.content_hash)

            logger.info(
                "Indexed %d chunk(s), skipped %d duplicate(s) for doc %s.",
                inserted, skipped, doc_id,
            )

            # -----------------------------------------------------------
            # 6. Mark indexed (text pipeline complete)
            # -----------------------------------------------------------
            await update_document_status(conn, doc_uuid, DocStatus.indexed)
            logger.info("Document %s text pipeline complete.", doc_id)

        except Exception as exc:
            logger.exception("Ingestion failed for doc %s: %s", doc_id, exc)
            await update_document_status(
                conn, doc_uuid, DocStatus.failed,
                error_message=str(exc)[:1024],
            )
            raise  # Let arq record the failure

    # -------------------------------------------------------------------
    # 7. Image extraction + captioning (PDF only, separate connection)
    #    Runs AFTER text indexing is complete and connection is released.
    #    Each image failure is isolated — document stays 'indexed'.
    # -------------------------------------------------------------------
    if doc.source_type != "pdf":
        return

    raw_path = Path(settings.raw_files_dir) / str(doc.tenant_id) / f"{doc_id}_{doc.filename}"
    logger.info("Starting image extraction for PDF doc %s", doc_id)

    extracted = extract_images_from_pdf(raw_path)
    if not extracted:
        logger.info("No images found/extracted from doc %s", doc_id)
        return

    logger.info("Found %d image(s) to process for doc %s", len(extracted), doc_id)
    embedder = ctx["embedder"]

    for img in extracted:
        image_id: uuid.UUID | None = None
        try:
            # a. Save to disk
            img_path = save_image(
                image_bytes=img.image_bytes,
                tenant_id=str(doc.tenant_id),
                doc_id=doc_id,
                page_number=img.page_number,
                image_index=img.image_index,
            )

            async with pool.acquire() as conn:
                await conn.execute("SELECT set_service_role_context()")

                # b. Find nearby text chunk for context linking
                nearby_chunk_id = await find_nearby_chunk_id(
                    conn, doc_uuid, img.page_number
                )

                # c. Insert stub row (no caption/embedding yet)
                image_id = await insert_image(
                    conn,
                    doc_id=doc_uuid,
                    tenant_id=doc.tenant_id,
                    page_number=img.page_number,
                    image_index=img.image_index,
                    storage_path=str(img_path),
                    bbox=img.bbox,
                    nearby_chunk_id=nearby_chunk_id,
                )

            logger.info(
                "Inserted image stub %s (page=%s, idx=%d)",
                image_id, img.page_number, img.image_index,
            )

            # d. VLM caption (optional — gracefully skipped if model not pulled)
            caption_text = await caption_image(img.image_bytes)
            if caption_text is None:
                logger.info(
                    "VLM captioning skipped for image %s (model not available or failed).",
                    image_id,
                )
                continue

            # e. Embed caption
            caption_vector = embedder.embed([caption_text])[0].tolist()

            # f. Update DB row with caption + embedding
            async with pool.acquire() as conn:
                await conn.execute("SELECT set_service_role_context()")
                await update_image_caption(conn, image_id, caption_text, caption_vector)

            logger.info(
                "Captioned + embedded image %s: '%.80s…'",
                image_id, caption_text,
            )

        except Exception as exc:
            logger.warning(
                "Failed to process image (page=%s, idx=%d, id=%s): %s",
                img.page_number, img.image_index, image_id, exc,
            )
            # Continue with next image — do not fail the whole document

    logger.info("Image pipeline complete for doc %s.", doc_id)
