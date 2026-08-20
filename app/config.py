from __future__ import annotations

import os
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    All configuration is read from environment variables (and .env file in dev).
    No secrets are hardcoded here — see .env.example for every required variable.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------
    database_url: str = Field(
        ...,
        description=(
            "asyncpg DSN. Example: "
            "postgresql+asyncpg://postgres:password@localhost:5432/postgres"
        ),
    )

    # ------------------------------------------------------------------
    # Redis / arq
    # ------------------------------------------------------------------
    redis_url: str = Field(
        default="redis://redis:6379",
        description="Redis DSN used by arq for the job queue.",
    )

    # ------------------------------------------------------------------
    # Embeddings — sentence-transformers (runs in-process, locally)
    # ------------------------------------------------------------------
    embedding_model: str = Field(
        default="BAAI/bge-large-en-v1.5",
        description=(
            "HuggingFace model name for sentence-transformers. "
            "Must produce 1024-dimensional embeddings to match the DB schema. "
            "Change embedding vector dimension in migrations if you swap models."
        ),
    )
    embedding_batch_size: int = Field(
        default=32,
        description="Batch size for encoding chunks during ingestion.",
    )
    embedding_device: Literal["cpu", "cuda", "mps"] = Field(
        default="cpu",
        description="Torch device for the embedding model.",
    )

    # ------------------------------------------------------------------
    # Generation — local Ollama (no external calls)
    # ------------------------------------------------------------------
    ollama_base_url: str = Field(
        default="http://ollama:11434",
        description="Base URL of the local Ollama server.",
    )
    ollama_model: str = Field(
        default="llama3.1:8b",
        description="Ollama model tag to use for generation.",
    )
    ollama_timeout: int = Field(
        default=120,
        description="HTTP timeout in seconds for Ollama generation calls.",
    )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    top_k: int = Field(
        default=8,
        description="Default number of chunks returned by similarity search (Phase 1 fallback).",
    )
    similarity_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Phase 1 cosine similarity gate threshold — kept for reference. "
            "Phase 2 uses GATE_RERANK_THRESHOLD instead."
        ),
    )

    # ------------------------------------------------------------------
    # Phase 2: Hybrid retrieval & reranking
    # ------------------------------------------------------------------
    rerank_model: str = Field(
        default="BAAI/bge-reranker-large",
        description=(
            "HuggingFace model name for the cross-encoder reranker "
            "(sentence_transformers.CrossEncoder). Runs locally in-process."
        ),
    )
    rerank_top_n: int = Field(
        default=6,
        description="Number of top-scored chunks after reranking to pass into the LLM context.",
    )
    hybrid_candidate_k: int = Field(
        default=40,
        description=(
            "Candidate pool size for each individual search (vector + FTS) "
            "before RRF fusion. Must be >= rerank_top_n."
        ),
    )
    rrf_k: int = Field(
        default=60,
        description=(
            "Reciprocal Rank Fusion smoothing constant. "
            "Higher values reduce the influence of top-ranked documents. "
            "Standard value from literature is 60."
        ),
    )
    gate_rerank_threshold: float = Field(
        default=0.0,
        description=(
            "Confidence gate: refuse if the top reranker score is below this value. "
            "CrossEncoder logits are unbounded (not [0,1]), so this threshold is "
            "corpus-dependent and MUST be tuned empirically before production use. "
            "The default 0.0 is a conservative floor (negative logit = clear irrelevance)."
        ),
    )
    mmr_enabled: bool = Field(
        default=False,
        description=(
            "Enable Maximal Marginal Relevance diversity pass after reranking. "
            "Reduces near-duplicate chunks from the same section in the LLM context. "
            "Toggle via env var for A/B testing against plain top-N."
        ),
    )
    mmr_lambda: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "MMR trade-off: 1.0 = pure relevance (no diversity), "
            "0.0 = maximum diversity (no relevance). 0.5 balances both."
        ),
    )

    # ------------------------------------------------------------------
    # File storage
    # ------------------------------------------------------------------
    raw_files_dir: str = Field(
        default="/app/data/raw",
        description="Host path (or Docker volume mount) where uploaded files are stored.",
    )

    # ------------------------------------------------------------------
    # Auth / Tenancy
    # ------------------------------------------------------------------
    # Phase 1: tenant_id is accepted in the request body.
    # Phase 2: replace with JWT middleware that extracts tenant_id from claims.
    default_tenant_id: str = Field(
        default="",
        description=(
            "Fallback tenant UUID for local dev. "
            "Must be a valid UUID. "
            "Leave empty to require tenant_id on every request."
        ),
    )

    # ------------------------------------------------------------------
    # Supabase / Postgres auth
    # ------------------------------------------------------------------
    supabase_service_role_key: str = Field(
        default="",
        description=(
            "Supabase service-role key. "
            "Used ONLY by the ingestion worker to bypass RLS. "
            "Never expose to API clients."
        ),
    )
    # The anon/public key is used by API endpoints (Phase 2 will verify JWTs).
    supabase_anon_key: str = Field(
        default="",
        description="Supabase anonymous/public key for client-facing operations.",
    )
    supabase_jwt_secret: str = Field(
        default="",
        description=(
            "JWT secret for verifying Supabase tokens. "
            "Required for real auth in Phase 2."
        ),
    )

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------
    chunk_target_tokens: int = Field(
        default=384,
        description="Target token count per chunk (fits within 256–512 range).",
    )
    chunk_overlap_tokens: int = Field(
        default=46,
        description="Overlap between consecutive chunks (~12% of 384).",
    )

    # ------------------------------------------------------------------
    # Generation / citation
    # ------------------------------------------------------------------
    max_context_chunks: int = Field(
        default=8,
        description="Maximum number of chunks to include in the LLM context window.",
    )
    citation_retry_enabled: bool = Field(
        default=True,
        description="Retry generation once with a stricter prompt if citation validation fails.",
    )

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    version: str = Field(
        default="2.0.0",
        description="Application version surfaced in /health and docs.",
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator("database_url")
    @classmethod
    def _strip_sync_prefix(cls, v: str) -> str:
        """Normalise SQLAlchemy-style DSNs to bare asyncpg DSNs."""
        if v.startswith("postgresql+asyncpg://"):
            return v.replace("postgresql+asyncpg://", "postgresql://", 1)
        return v


# Module-level singleton — import this everywhere.
settings = Settings()
