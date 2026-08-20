"""
app/worker/worker_settings.py

arq WorkerSettings — defines the job queue, Redis connection,
and the startup/shutdown lifecycle (db pool, embedder, chunker).
"""
from __future__ import annotations

import logging

import asyncpg
from arq import create_pool
from arq.connections import RedisSettings

from app.config import settings
from app.services.embedder import load_embedder
from app.worker.chunker import StructureAwareChunker
from app.worker.tasks import ingest_document

logger = logging.getLogger(__name__)


async def startup(ctx: dict) -> None:
    """Initialise shared resources injected into every job's ctx dict."""
    logger.info("Worker starting up…")

    async def _init_conn(conn) -> None:
        import json as _json
        await conn.set_type_codec(
            "doc_status", encoder=str, decoder=str, schema="public", format="text"
        )
        await conn.set_type_codec(
            "jsonb", encoder=_json.dumps, decoder=_json.loads,
            schema="pg_catalog", format="text",
        )

    ctx["db_pool"] = await asyncpg.create_pool(
        dsn=settings.database_url,
        min_size=2,
        max_size=10,
        command_timeout=30,
        init=_init_conn,
    )
    logger.info("DB pool created.")

    # Embedding model (loads weights from cache or downloads once)
    ctx["embedder"] = load_embedder()
    logger.info("Embedder ready.")

    # Chunker (stateless, but we construct once to pay tiktoken init cost)
    ctx["chunker"] = StructureAwareChunker()
    logger.info("Chunker ready.")


async def shutdown(ctx: dict) -> None:
    """Clean up on worker shutdown."""
    if "db_pool" in ctx:
        await ctx["db_pool"].close()
        logger.info("DB pool closed.")


class WorkerSettings:
    """
    arq WorkerSettings class.
    arq discovers this class via the module path:
        python -m arq app.worker.worker_settings.WorkerSettings
    """
    functions = [ingest_document]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(settings.redis_url)

    # Retry failed jobs up to 3 times with exponential back-off
    max_tries = 3
    job_timeout = 600  # 10 minutes per job max

    # Keep job results in Redis for 24 hours for debugging
    keep_result = 86_400
