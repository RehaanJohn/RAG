-- migrations/003_phase4_images.sql
-- Phase 4: images table with FTS + vector index + RLS.
-- IDEMPOTENT — safe to run multiple times.

-- ---------------------------------------------------------------------------
-- 1. Images table
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS images (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id          uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    tenant_id       uuid NOT NULL,
    page_number     int,
    image_index     int  NOT NULL,      -- 0-based within the document
    storage_path    text NOT NULL,      -- absolute path on disk inside the container
    bbox            jsonb,              -- {x0,y0,x1,y1} in points, nullable
    nearby_chunk_id uuid REFERENCES chunks(id) ON DELETE SET NULL,
    caption         text,              -- NULL until VLM captioning completes
    -- Generated FTS column: empty caption produces empty tsvector (no FTS hits)
    caption_tsv     tsvector GENERATED ALWAYS AS
                        (to_tsvector('english', coalesce(caption, ''))) STORED,
    embedding       vector(1024),      -- NULL until caption embedded; same dim as chunks
    created_at      timestamptz NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- 2. Indexes
-- ---------------------------------------------------------------------------

-- FTS on caption
CREATE INDEX IF NOT EXISTS idx_images_caption_tsv
    ON images USING gin(caption_tsv);

-- HNSW for fast approximate nearest-neighbour on caption embedding
-- Only rows with non-NULL embedding participate in vector search
CREATE INDEX IF NOT EXISTS idx_images_embedding
    ON images USING hnsw(embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_images_doc_id
    ON images (doc_id);

CREATE INDEX IF NOT EXISTS idx_images_tenant_id
    ON images (tenant_id);

-- ---------------------------------------------------------------------------
-- 3. Row-Level Security (same pattern as chunks)
-- ---------------------------------------------------------------------------

ALTER TABLE images ENABLE ROW LEVEL SECURITY;

DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE tablename = 'images' AND policyname = 'tenant_isolation_images'
    ) THEN
        CREATE POLICY tenant_isolation_images
            ON images FOR ALL
            USING (tenant_id = current_setting('app.tenant_id', true)::uuid);
    END IF;
END $$;

DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE tablename = 'images' AND policyname = 'service_role_bypass_images'
    ) THEN
        CREATE POLICY service_role_bypass_images
            ON images FOR ALL
            TO service_role
            USING (true);
    END IF;
END $$;
