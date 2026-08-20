from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

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
    """A single retrieved chunk with its similarity or fusion score."""
    id: uuid.UUID
    doc_id: uuid.UUID
    content: str
    metadata: dict[str, Any]
    score: float         # cosine similarity [0,1] from vector search, or RRF fused score
    rerank_score: float | None = None  # CrossEncoder logit (unbounded); None before reranking

    # Convenience: citation ID used in LLM context and returned to client
    @property
    def citation_id(self) -> str:
        chunk_index = self.metadata.get("chunk_index", 0)
        return f"{self.doc_id}:{chunk_index}"


# ---------------------------------------------------------------------------
# Source object — resolved citation, returned to the client
# ---------------------------------------------------------------------------


class SourceObject(BaseModel):
    """A resolved, structured citation returned alongside the answer."""
    citation_id: str           # e.g. "doc_uuid:3"
    filename: str
    page_number: int | None = None
    section_path: str | None = None
    score: float               # similarity score for transparency


# ---------------------------------------------------------------------------
# Query schemas
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    tenant_id: uuid.UUID = Field(
        ...,
        description=(
            "Tenant UUID. Phase 1: supplied in request body. "
            "Phase 2: extracted from JWT."
        ),
    )
    top_k: int = Field(default=8, ge=1, le=32)


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


# ---------------------------------------------------------------------------
# Error response
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    detail: str
    code: str | None = None
