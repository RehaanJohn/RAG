-- =============================================================================
-- Phase 1 RAG System — Initial Schema Migration
-- =============================================================================
-- Run against the self-hosted Supabase Postgres instance.
-- Supabase already enables the `vector` extension in most images; the line
-- below is idempotent and safe to re-run.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 1. Extensions
-- ---------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- 2. Custom types
-- ---------------------------------------------------------------------------
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'doc_status') THEN
    CREATE TYPE doc_status AS ENUM (
      'pending',
      'parsing',
      'chunking',
      'embedding',
      'indexed',
      'failed'
    );
  END IF;
END
$$;

-- ---------------------------------------------------------------------------
-- 3. documents
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
  id           UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  filename     TEXT        NOT NULL,
  source_type  TEXT        NOT NULL CHECK (source_type IN ('pdf', 'txt', 'markdown')),
  status       doc_status  NOT NULL DEFAULT 'pending',
  tenant_id    UUID        NOT NULL,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  -- error_message is populated when status = 'failed'
  error_message TEXT,
  metadata     JSONB       NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_documents_tenant_id ON documents (tenant_id);
CREATE INDEX IF NOT EXISTS idx_documents_status    ON documents (status);

-- ---------------------------------------------------------------------------
-- 4. chunks
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunks (
  id           UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  doc_id       UUID        NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
  content      TEXT        NOT NULL,
  -- SHA-256 hex digest of content, used for deduplication
  content_hash TEXT        NOT NULL,
  -- bge-large-en-v1.5 produces 1024-dimensional embeddings
  embedding    vector(1024),
  -- metadata carries: section_path, page_number, chunk_index, parent_chunk_id
  metadata     JSONB       NOT NULL DEFAULT '{}',
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Deduplication: inserting a chunk with the same hash is a no-op
CREATE UNIQUE INDEX IF NOT EXISTS uidx_chunks_content_hash ON chunks (content_hash);

-- Foreign-key lookup index
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks (doc_id);

-- HNSW index for approximate nearest-neighbour cosine search.
-- m=16 and ef_construction=64 are solid Phase-1 defaults; tune later.
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
  ON chunks
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- ---------------------------------------------------------------------------
-- 5. query_logs
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS query_logs (
  id                   UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  -- The raw user query
  query                TEXT        NOT NULL,
  tenant_id            UUID,
  -- Array of chunk UUIDs returned by retrieval, stored as JSONB
  retrieved_chunk_ids  JSONB,
  top_similarity_score FLOAT,
  answer               TEXT,
  was_refused          BOOLEAN     NOT NULL DEFAULT false,
  latency_ms           INT,
  created_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_query_logs_tenant_id  ON query_logs (tenant_id);
CREATE INDEX IF NOT EXISTS idx_query_logs_created_at ON query_logs (created_at DESC);

-- ---------------------------------------------------------------------------
-- 6. Row-Level Security
-- ---------------------------------------------------------------------------

-- documents ----------------------------------------------------------------
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

-- Normal tenant access: session setting must match row's tenant_id.
CREATE POLICY tenant_isolation_documents
  ON documents
  FOR ALL
  USING (
    tenant_id = current_setting('app.tenant_id', true)::UUID
  );

-- Service-role bypass (used exclusively by the ingestion worker).
-- The worker sets: SET LOCAL app.role = 'service_role'
CREATE POLICY service_role_bypass_documents
  ON documents
  FOR ALL
  USING (
    current_setting('app.role', true) = 'service_role'
  );

-- chunks -------------------------------------------------------------------
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation_chunks
  ON chunks
  FOR ALL
  USING (
    doc_id IN (
      SELECT id FROM documents
      WHERE tenant_id = current_setting('app.tenant_id', true)::UUID
    )
  );

CREATE POLICY service_role_bypass_chunks
  ON chunks
  FOR ALL
  USING (
    current_setting('app.role', true) = 'service_role'
  );

-- query_logs ---------------------------------------------------------------
-- No RLS on query_logs — service role writes, only internal reads.
-- Restrict at application layer if multi-tenant log isolation is needed.

-- ---------------------------------------------------------------------------
-- 7. Helper function: set_tenant_context
-- ---------------------------------------------------------------------------
-- Call this at the start of every DB connection/transaction when acting on
-- behalf of a tenant (API path). The worker calls set_service_role_context().
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION set_tenant_context(p_tenant_id UUID)
RETURNS void
LANGUAGE plpgsql
AS $$
BEGIN
  PERFORM set_config('app.tenant_id', p_tenant_id::TEXT, true);
  PERFORM set_config('app.role', '', true);
END;
$$;

CREATE OR REPLACE FUNCTION set_service_role_context()
RETURNS void
LANGUAGE plpgsql
AS $$
BEGIN
  PERFORM set_config('app.role', 'service_role', true);
END;
$$;
