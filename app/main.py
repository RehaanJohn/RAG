"""
app/main.py

FastAPI application entry point.

Lifecycle:
  startup:  load embedding model, create asyncpg pool, run DB migrations
  shutdown: close DB pool

All heavy initialisation happens once in the lifespan context manager so
every request after the first benefits from warm resources.
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import asyncpg
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.routers import documents, query
from app.services.embedder import load_embedder
from app.services.reranker import load_reranker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)


# ---------------------------------------------------------------------------
# lifespan context manager
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Runs at startup and shutdown.
    All resources stored on app.state are available via request.app.state
    in dependency functions.
    """
    logger.info("=== RAG API starting up ===")

    # 1. Ensure raw file storage directory exists
    Path(settings.raw_files_dir).mkdir(parents=True, exist_ok=True)

    # 2. Load embedding model (downloads on first start, cached in hf_cache volume)
    logger.info("Loading embedding model: %s", settings.embedding_model)
    app.state.embedder = load_embedder()
    logger.info("Embedding model ready. dim=%d", app.state.embedder.embedding_dim)

    # 3. Load cross-encoder reranker (Phase 2 — downloads on first start)
    logger.info("Loading reranker model: %s", settings.rerank_model)
    app.state.reranker = load_reranker()
    logger.info("Reranker ready: %s", app.state.reranker.model_name)

    # 4. Create asyncpg connection pool
    logger.info("Connecting to database…")

    async def _init_conn(conn: asyncpg.Connection) -> None:
        """Register type codecs on every new connection in the pool."""
        import json as _json

        # Decode the custom doc_status enum as a plain str
        await conn.set_type_codec(
            "doc_status",
            encoder=str,
            decoder=str,
            schema="public",
            format="text",
        )
        # Decode JSONB as Python dicts via json.loads
        await conn.set_type_codec(
            "jsonb",
            encoder=_json.dumps,
            decoder=_json.loads,
            schema="pg_catalog",
            format="text",
        )

    app.state.db_pool = await asyncpg.create_pool(
        dsn=settings.database_url,
        min_size=2,
        max_size=20,
        command_timeout=30,
        init=_init_conn,
    )
    logger.info("Database pool ready.")

    # 5. Apply Phase 1 + Phase 2 migrations (idempotent SQL)
    migrations_dir = Path(__file__).parent.parent / "migrations"
    for migration_name in ["001_init.sql", "002_phase2.sql"]:
        migration_file = migrations_dir / migration_name
        if migration_file.exists():
            logger.info("Applying migration: %s", migration_name)
            sql = migration_file.read_text()
            async with app.state.db_pool.acquire() as conn:
                try:
                    await conn.execute(sql)
                    logger.info("Migration applied (or already current): %s", migration_name)
                except Exception as exc:
                    logger.warning("Migration warning (likely already applied): %s", exc)
        else:
            logger.warning("Migration file not found — skipping: %s", migration_name)

    logger.info("=== RAG API ready ===")

    yield  # Application runs here

    # Shutdown
    logger.info("=== RAG API shutting down ===")
    await app.state.db_pool.close()
    logger.info("Database pool closed.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG API",
    description=(
        "Phase 2 — Hybrid Search + Local Reranker. "
        "Vector + FTS retrieval, RRF fusion, CrossEncoder reranking. "
        "All models run locally. No external API calls."
    ),
    version=settings.version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — restrict in production via ALLOWED_ORIGINS env var
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(documents.router)
app.include_router(query.router)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["meta"])
async def health() -> JSONResponse:
    """Simple liveness probe used by Docker Compose healthchecks."""
    return JSONResponse({"status": "ok", "version": settings.version})


@app.get("/", tags=["meta"])
async def root() -> JSONResponse:
    return JSONResponse(
        {
            "service": "RAG API",
            "phase": 2,
            "docs": "/docs",
            "health": "/health",
        }
    )
