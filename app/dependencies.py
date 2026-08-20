"""
app/dependencies.py

FastAPI dependency providers for shared resources:
  - get_db()       — asyncpg connection from the connection pool
  - get_db_pool()  — the pool itself (for acquiring multiple connections)
  - get_embedder() — EmbeddingService singleton
  - get_reranker() — RerankerService singleton (Phase 2)

The pool, embedder, and reranker are initialised in app/main.py via the
lifespan context manager, then stored in app.state for retrieval here.
"""
from __future__ import annotations

from typing import AsyncGenerator

import asyncpg
from fastapi import Request

from app.services.embedder import EmbeddingService
from app.services.reranker import RerankerService


async def get_db(request: Request) -> AsyncGenerator[asyncpg.Connection, None]:
    """
    Yield a single asyncpg connection from the pool, then release it.
    All callers receive their own connection per request.
    """
    pool: asyncpg.Pool = request.app.state.db_pool
    async with pool.acquire() as conn:
        yield conn


def get_db_pool(request: Request) -> asyncpg.Pool:
    """Return the raw connection pool. Use when you need multiple connections."""
    return request.app.state.db_pool


def get_embedder(request: Request) -> EmbeddingService:
    """Return the application-level EmbeddingService singleton."""
    return request.app.state.embedder


def get_reranker(request: Request) -> RerankerService:
    """Return the application-level RerankerService singleton (Phase 2)."""
    return request.app.state.reranker
