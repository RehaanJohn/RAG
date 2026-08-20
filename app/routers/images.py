"""
app/routers/images.py

Phase 4 image endpoints:
  - GET  /images/{image_id}/file — serve raw image file
  - POST /images/{image_id}/ask  — interactive direct visual Q&A on a specific image
"""
from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import FileResponse

from app.config import settings
from app.dependencies import get_db
from app.models.db import get_image
from app.models.schemas import AskImageRequest, AskImageResponse, ErrorResponse
from app.services.image_store import load_image
from app.services.vlm import ask_image

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/images", tags=["images"])


@router.get(
    "/{image_id}/file",
    summary="Download or view raw image file.",
    responses={404: {"model": ErrorResponse}},
)
async def get_image_file(
    image_id: uuid.UUID,
    tenant_id: uuid.UUID = Query(..., description="Tenant UUID for access control"),
    conn: asyncpg.Connection = Depends(get_db),
) -> FileResponse:
    """Serve the extracted image file from disk after verifying tenant ownership."""
    await conn.execute("SELECT set_tenant_context($1::uuid)", tenant_id)
    image = await get_image(conn, image_id, tenant_id)
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image {image_id} not found or tenant mismatch.",
        )

    file_path = Path(image.storage_path)
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image file missing from storage.",
        )

    return FileResponse(
        path=str(file_path),
        media_type="image/png",
        filename=file_path.name,
    )


@router.post(
    "/{image_id}/ask",
    response_model=AskImageResponse,
    summary="Direct interactive visual Q&A on a specific image.",
    responses={404: {"model": ErrorResponse}},
)
async def ask_image_endpoint(
    image_id: uuid.UUID,
    request: AskImageRequest,
    raw_request: Request,
    conn: asyncpg.Connection = Depends(get_db),
) -> AskImageResponse:
    """
    Direct visual Q&A with the local VLM against an extracted image.
    Bypasses text retrieval — passes the raw image bytes and question to the VLM.
    """
    t_start = time.monotonic()

    # 1. Enforce tenant ownership
    await conn.execute("SELECT set_tenant_context($1::uuid)", request.tenant_id)
    image = await get_image(conn, image_id, request.tenant_id)
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image {image_id} not found or tenant mismatch.",
        )

    # 2. Load image bytes from storage
    try:
        image_bytes = load_image(image.storage_path)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image file not found on disk storage.",
        )

    # 3. Call local VLM
    answer, vlm_unavailable = await ask_image(image_bytes, request.question)
    latency_ms = int((time.monotonic() - t_start) * 1000)

    # 4. Track image in conversation history if conversation_id provided
    if request.conversation_id:
        try:
            redis = raw_request.app.state.redis
            key = f"img_history:{request.conversation_id}"
            await redis.lpush(key, str(image_id))
            await redis.ltrim(key, 0, 4)  # keep last 5
            await redis.expire(key, 3600) # 1-hour TTL
        except Exception as exc:
            logger.warning("Failed to update conversation image history in Redis: %s", exc)

    return AskImageResponse(
        image_id=image_id,
        question=request.question,
        answer=answer,
        conversation_id=request.conversation_id,
        latency_ms=latency_ms,
        vlm_unavailable=vlm_unavailable,
    )
