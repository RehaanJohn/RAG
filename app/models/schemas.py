from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DocStatus(str, Enum):
    pending = "pending"
    parsing = "parsing"
    chunking = "chunking"
    embedding = "embedding"
    indexed = "indexed"
    failed = "failed"


# ---------------------------------------------------------------------------
# Document schemas
# ---------------------------------------------------------------------------


class DocumentCreate(BaseModel):
    """Internal: created after a successful file upload."""
    filename: str
    source_type: str
    tenant_id: uuid.UUID
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentResponse(BaseModel):
    id: uuid.UUID
    filename: str
    source_type: str
    status: DocStatus
    tenant_id: uuid.UUID
    created_at: datetime
    error_message: str | None = None
    metadata: dict[str, Any]

    model_config = {"from_attributes": True}


class DocumentUploadResponse(BaseModel):
    doc_id: uuid.UUID
    status: DocStatus
    message: str = "Document queued for ingestion."


# ---------------------------------------------------------------------------
# Chunk schemas (internal — not returned to clients directly)
# ---------------------------------------------------------------------------


class ChunkResult(BaseModel):
    """A single retrieved text chunk with its similarity or fusion score."""
    id: uuid.UUID
    doc_id: uuid.UUID
    content: str
    metadata: dict[str, Any]
    score: float         # cosine similarity [0,1] from vector search, or RRF fused score
    rerank_score: float | None = None  # CrossEncoder logit (unbounded); None before reranking

    @property
    def citation_id(self) -> str:
        chunk_index = self.metadata.get("chunk_index", 0)
        return f"{self.doc_id}:{chunk_index}"

    @property
    def rerank_content(self) -> str:
        """Text used by the cross-encoder reranker."""
        return self.content


class ImageResult(BaseModel):
    """A single retrieved image with its similarity or fusion score."""
    id: uuid.UUID
    doc_id: uuid.UUID
    page_number: int | None
    image_index: int
    storage_path: str
    caption: str | None = None       # None if VLM captioning not yet completed
    score: float                     # RRF fused score or cosine similarity
    rerank_score: float | None = None

    @property
    def citation_id(self) -> str:
        return f"{self.doc_id}:img_{self.image_index}"

    @property
    def rerank_content(self) -> str:
        """Caption text used by the cross-encoder reranker."""
        return self.caption or ""


# ---------------------------------------------------------------------------
# Source object — resolved citation, returned to the client
# ---------------------------------------------------------------------------


class SourceObject(BaseModel):
    """A resolved, structured citation returned alongside the answer."""
    type: Literal["text", "image"] = "text"
    citation_id: str           # e.g. "doc_uuid:3" or "doc_uuid:img_1"
    filename: str
    page_number: int | None = None
    section_path: str | None = None
    score: float               # retrieval score for transparency
    image_url: str | None = None   # populated for type="image" entries


# ---------------------------------------------------------------------------
# Query schemas
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    tenant_id: uuid.UUID = Field(
        ...,
        description=(
            "Tenant UUID. Phase 1: supplied in request body. "
            "Phase 3: extracted from JWT."
        ),
    )
    top_k: int = Field(default=8, ge=1, le=32)
    conversation_id: uuid.UUID | None = Field(
        default=None,
        description=(
            "Optional session UUID. When provided, recently-shown images are "
            "tracked in Redis so follow-up visual Q&A can be auto-routed."
        ),
    )


class QueryResponse(BaseModel):
    query: str
    answer: str
    was_refused: bool
    top_similarity_score: float | None = None
    sources: list[SourceObject] = Field(default_factory=list)
    latency_ms: int
    # Phase 2 additions (None for refused queries)
    rerank_scores: dict[str, float] | None = None
    """Map of chunk_id → reranker logit for every candidate passed to the LLM."""
    retrieval_candidate_count: int | None = None
    """Total unique candidates from RRF fusion before reranking."""
    # Phase 4 additions
    routed_to_image_qa: bool = False
    """True when the query was classified as referring to a previously-shown image."""


class AskImageRequest(BaseModel):
    """Request body for POST /images/{image_id}/ask."""
    question: str = Field(..., min_length=1, max_length=2048)
    tenant_id: uuid.UUID
    conversation_id: uuid.UUID | None = None


class AskImageResponse(BaseModel):
    """Response from POST /images/{image_id}/ask."""
    image_id: uuid.UUID
    question: str
    answer: str
    conversation_id: uuid.UUID | None = None
    latency_ms: int
    vlm_unavailable: bool = False
    """True when VLM model is not loaded — answer will be empty."""


# ---------------------------------------------------------------------------
# Error response
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    detail: str
    code: str | None = None
