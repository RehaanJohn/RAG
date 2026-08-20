"""
app/routers/documents.py

POST /documents — upload a file and enqueue it for ingestion
GET  /documents/{doc_id} — poll ingestion status
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

import asyncpg
from arq import create_pool
from arq.connections import RedisSettings
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.config import settings
from app.dependencies import get_db
from app.models.db import create_document, get_document
from app.models.schemas import DocStatus, DocumentResponse, DocumentUploadResponse, ErrorResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

# Allowed MIME types and their canonical source_type names
_ALLOWED_TYPES: dict[str, str] = {
    "application/pdf": "pdf",
    "text/plain": "txt",
    "text/markdown": "markdown",
    "text/x-markdown": "markdown",
    # Browsers sometimes send these for .md files
    "application/octet-stream": None,  # resolved by extension below
}

_EXTENSION_MAP: dict[str, str] = {
    ".pdf": "pdf",
    ".txt": "txt",
    ".md": "markdown",
    ".markdown": "markdown",
}


def _resolve_source_type(filename: str, content_type: str) -> str:
    """Determine source_type from MIME type, falling back to file extension."""
    ct = (content_type or "").split(";")[0].strip().lower()
    if ct in _ALLOWED_TYPES and _ALLOWED_TYPES[ct] is not None:
        return _ALLOWED_TYPES[ct]
    suffix = Path(filename).suffix.lower()
    if suffix in _EXTENSION_MAP:
        return _EXTENSION_MAP[suffix]
    raise HTTPException(
        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        detail=f"Unsupported file type. Allowed: PDF, TXT, Markdown. Got: {content_type!r} / {suffix!r}",
    )


# ---------------------------------------------------------------------------
# POST /documents
# ---------------------------------------------------------------------------

@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=DocumentUploadResponse,
    responses={415: {"model": ErrorResponse}},
)
async def upload_document(
    file: UploadFile = File(..., description="PDF, TXT, or Markdown file to ingest."),
    tenant_id: uuid.UUID = Form(
        ...,
        description="Tenant UUID. Phase 1: supplied in form data. Phase 2: from JWT.",
    ),
    conn: asyncpg.Connection = Depends(get_db),
) -> DocumentUploadResponse:
    """
    Accept a file upload, persist it to disk, create a documents row,
    and enqueue an arq ingestion job.
    """
    filename = file.filename or "unnamed"
    source_type = _resolve_source_type(filename, file.content_type or "")

    # -----------------------------------------------------------------------
    # 1. Save raw file
    # -----------------------------------------------------------------------
    tenant_dir = Path(settings.raw_files_dir) / str(tenant_id)
    tenant_dir.mkdir(parents=True, exist_ok=True)

    doc_id = uuid.uuid4()
    safe_filename = f"{doc_id}_{filename}"
    dest_path = tenant_dir / safe_filename

    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    dest_path.write_bytes(content)
    logger.info("Saved upload: %s (%d bytes)", dest_path, len(content))

    # -----------------------------------------------------------------------
    # 2. Create DB row (status = pending)
    # -----------------------------------------------------------------------
    created_id = await create_document(
        conn,
        filename=filename,
        source_type=source_type,
        tenant_id=tenant_id,
        metadata={"file_size_bytes": len(content)},
    )
    # Use the ID assigned by the DB (which may differ if we didn't pre-set it)
    doc_id = created_id

    # Rename saved file to match DB-assigned ID
    final_path = tenant_dir / f"{doc_id}_{filename}"
    dest_path.rename(final_path)

    # -----------------------------------------------------------------------
    # 3. Enqueue arq job
    # -----------------------------------------------------------------------
    redis_pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    await redis_pool.enqueue_job("ingest_document", str(doc_id))
    await redis_pool.aclose()

    logger.info("Enqueued ingestion job for doc_id=%s", doc_id)

    return DocumentUploadResponse(
        doc_id=doc_id,
        status=DocStatus.pending,
        message="Document received and queued for ingestion.",
    )


# ---------------------------------------------------------------------------
# GET /documents/{doc_id}
# ---------------------------------------------------------------------------

@router.get(
    "/{doc_id}",
    response_model=DocumentResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_document_status(
    doc_id: uuid.UUID,
    conn: asyncpg.Connection = Depends(get_db),
) -> DocumentResponse:
    """Poll the ingestion status of a document."""
    doc = await get_document(conn, doc_id)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {doc_id} not found.",
        )
    return doc
