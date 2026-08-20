# RAG System — Phase 1

A production-grade, **fully local** Retrieval-Augmented Generation pipeline.
Every model runs on your own hardware. No data leaves your infrastructure.

| Component | Technology |
|---|---|
| API | FastAPI + asyncpg |
| Database | Self-hosted Supabase (Postgres 15 + pgvector) |
| Embeddings | `sentence-transformers` (in-process, CPU/GPU) |
| Generation | Local Ollama (`llama3.1:8b` by default) |
| Background jobs | `arq` (Redis-backed) |
| Containerisation | Docker Compose |

---

## Quick Start

### 1. Prerequisites

- Docker ≥ 24 and Docker Compose v2
- ~10 GB free disk space (Ollama model ~4.7 GB + Postgres data + image layers)
- No GPU required (CPU-only by default)

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — at minimum set a real SUPABASE_JWT_SECRET (32+ chars)
# All other defaults work for local dev
```

### 3. Start the stack

```bash
docker compose up -d --build
```

This will:
1. Build the FastAPI app and worker images (downloads embedding model weights into the image)
2. Start Postgres, Redis, and Ollama
3. Run `ollama pull llama3.1:8b` (~4.7 GB download on first run)
4. Apply the database migration (`migrations/001_init.sql`)
5. Start the API on `http://localhost:8000`

**First start takes 5–15 minutes** while the Ollama model downloads.

Monitor progress:
```bash
docker compose logs -f
docker compose logs -f ollama-init   # watch model pull progress
```

### 4. Verify

```bash
curl http://localhost:8000/health
# → {"status":"ok","version":"1.0.0"}
```

- **API docs**: http://localhost:8000/docs
- **Supabase Studio**: http://localhost:3000

---

## API Reference

### Upload a document

```bash
curl -X POST http://localhost:8000/documents \
  -F "file=@/path/to/document.pdf" \
  -F "tenant_id=<your-uuid>"
```

**Response** (202 Accepted):
```json
{
  "doc_id": "3f2504e0-...",
  "status": "pending",
  "message": "Document received and queued for ingestion."
}
```

Supported file types: `.pdf`, `.txt`, `.md`

### Poll ingestion status

```bash
curl http://localhost:8000/documents/<doc_id>
```

Status progression: `pending → parsing → chunking → embedding → indexed`
(or `failed` with `error_message` on failure)

### Query the knowledge base

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is retrieval-augmented generation?",
    "tenant_id": "<your-uuid>",
    "top_k": 8
  }'
```

**Response**:
```json
{
  "query": "What is retrieval-augmented generation?",
  "answer": "RAG is a technique that... [doc_uuid:2] ...",
  "was_refused": false,
  "top_similarity_score": 0.847,
  "sources": [
    {
      "citation_id": "doc_uuid:2",
      "filename": "rag_overview.md",
      "page_number": null,
      "section_path": "What is RAG?",
      "score": 0.847
    }
  ],
  "latency_ms": 1423
}
```

If the query doesn't match any indexed content (score < threshold):
```json
{
  "was_refused": true,
  "answer": "I couldn't find anything in the knowledge base that answers this. Try rephrasing, or this may be outside what's currently indexed.",
  "sources": []
}
```

---

## Smoke Tests

Run the end-to-end smoke test suite against a running stack:

```bash
# Install test deps (httpx)
pip install httpx

# Run tests (generates a random tenant_id by default)
python smoke_tests/run_smoke_tests.py

# Use a specific tenant_id
python smoke_tests/run_smoke_tests.py --tenant-id <uuid>

# Against a remote deployment
python smoke_tests/run_smoke_tests.py --base-url http://your-server:8000
```

The script:
1. Uploads sample documents from `smoke_tests/sample_docs/`
2. Waits for indexing to complete (up to 5 minutes)
3. Runs 5 in-domain queries — asserts `was_refused=False` and `sources` non-empty
4. Runs 3 out-of-domain queries — asserts `was_refused=True`

---

## Swapping Models

### Change the embedding model

> ⚠️ Changing the embedding model requires re-indexing all existing documents
> (embeddings are not cross-model comparable) and updating the `vector(1024)`
> dimension in the migration if the new model has a different output size.

1. Edit `.env`:
   ```
   EMBEDDING_MODEL=intfloat/e5-large-v2   # also 1024-dim, compatible
   ```
2. If the new model has a different dimension, update `migrations/001_init.sql`:
   ```sql
   -- Change vector(1024) → vector(768) for base-sized models
   ```
3. Rebuild:
   ```bash
   docker compose build --build-arg EMBEDDING_MODEL=intfloat/e5-large-v2 app worker
   docker compose up -d
   ```

### Change the generation model

1. Edit `.env`:
   ```
   OLLAMA_MODEL=mistral:7b
   ```
2. Restart the stack. The `ollama-init` service will pull the new model:
   ```bash
   docker compose up -d
   docker compose logs -f ollama-init
   ```

Popular alternatives:
| Model | Size | Notes |
|---|---|---|
| `llama3.1:8b` | 4.7 GB | Default; strong general reasoning |
| `mistral:7b` | 4.1 GB | Fast, good instruction following |
| `phi3:mini` | 2.3 GB | Lightweight, good for CPU |
| `gemma2:9b` | 5.5 GB | Google's model, strong performance |

---

## Environment Variables

See [`.env.example`](.env.example) for a fully documented list. Key variables:

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | — | asyncpg Postgres DSN |
| `EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | Local sentence-transformers model |
| `OLLAMA_MODEL` | `llama3.1:8b` | Local Ollama generation model |
| `SIMILARITY_THRESHOLD` | `0.5` | Gate threshold; below this → refusal |
| `TOP_K` | `8` | Default retrieval k |
| `CHUNK_TARGET_TOKENS` | `384` | Target tokens per chunk |
| `SUPABASE_SERVICE_ROLE_KEY` | — | Worker-only; never expose to clients |

---

## Architecture

```
Client
  │
  ▼
FastAPI (app)
  ├── POST /documents ──► Save file ──► Create DB row ──► Enqueue arq job
  │                                                              │
  │                                                              ▼
  │                                                       arq Worker
  │                                                         Parse → Chunk
  │                                                         Embed (local)
  │                                                         Index → pgvector
  │
  └── POST /query ──► Embed query (local)
                    ──► pgvector cosine search
                    ──► Confidence gate (score < threshold → refusal)
                    ──► Context assembly + citation tags
                    ──► Ollama local LLM
                    ──► Citation validation (uncited claims + hallucination check)
                    ──► Retry once if invalid
                    ──► Refusal fallback if retry fails
                    ──► Log to query_logs
                    ──► Return {answer, sources, was_refused}
```

---

## Security Notes

- **Service-role key** (`SUPABASE_SERVICE_ROLE_KEY`) bypasses Row-Level Security.
  It is used exclusively by the ingestion worker. Never include it in frontend code
  or expose it to API clients.
- **RLS** is enabled on `documents` and `chunks`. All tenant queries run in
  tenant context (`app.tenant_id` session variable set per request).
- **No external network calls**: all model inference is local. Verify this in
  production by inspecting `docker compose logs` for unexpected outbound connections.
- **Phase 2 auth**: the `tenant_id`-in-request-body pattern is intentionally
  simplified for Phase 1. Replace with JWT middleware before production deployment.

---

## What's NOT in Phase 1

These features are reserved for later phases:
- Hybrid keyword + semantic search (BM25 + pgvector RRF)
- Cross-encoder reranker
- Query transformation (HyDE, multi-query expansion)
- Semantic cache
- Real JWT authentication middleware
