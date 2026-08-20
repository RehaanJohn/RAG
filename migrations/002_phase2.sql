-- migrations/002_phase2.sql
-- Phase 2: Full-text search column + GIN index + extended query_logs.
--
-- This migration is IDEMPOTENT — safe to run multiple times.
-- `GENERATED ALWAYS AS ... STORED` backfills content_tsv for existing chunks
-- automatically when the column is added.

-- ---------------------------------------------------------------------------
-- 1. Full-text search: generated tsvector column on chunks
-- ---------------------------------------------------------------------------

ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS content_tsv
        tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

-- ---------------------------------------------------------------------------
-- 2. GIN index for fast full-text search
-- ---------------------------------------------------------------------------

CREATE INDEX IF NOT EXISTS idx_chunks_content_tsv
    ON chunks USING gin(content_tsv);

-- ---------------------------------------------------------------------------
-- 3. Extend query_logs with Phase 2 retrieval metadata
-- ---------------------------------------------------------------------------

ALTER TABLE query_logs
    ADD COLUMN IF NOT EXISTS rerank_scores            jsonb,
    ADD COLUMN IF NOT EXISTS retrieval_candidate_count int;

-- rerank_scores stores {chunk_id: score} for every candidate that passed RRF
-- retrieval_candidate_count is the total unique chunks from both searches before reranking
