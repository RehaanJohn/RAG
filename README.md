# RAG System — Phase 4 Multimodal Image Support (Local VLM)

A production-grade, **fully local and self-hosted** Retrieval-Augmented Generation pipeline.
Every model (embeddings, cross-encoder reranker, text generation, and vision-language model) runs on your own hardware. **No data ever leaves your infrastructure.**

| Component | Technology | Model / Spec |
|---|---|---|
| API | FastAPI + asyncpg | Async throughout |
| Database | Self-hosted Supabase | Postgres 15 + pgvector + GIN FTS |
| Embeddings | `sentence-transformers` | `BAAI/bge-large-en-v1.5` (1024-dim, in-process) |
| Reranker | `sentence-transformers` CrossEncoder | `BAAI/bge-reranker-large` (in-process) |
| Generation | Local Ollama | `llama3.1:8b` (configurable) |
| Vision VLM | Local Ollama | `llama3.2-vision` (configurable, optional) |
| Background jobs | `arq` + Redis | Async document parsing, chunking, VLM captioning |
| Containerisation | Docker Compose | Multi-container local deployment |

---

## What's New in Phase 4: Multimodal Images

1. **PDF Image Extraction & VLM Captioning**: Embedded diagrams and images are automatically extracted using PyMuPDF during background ingestion, saved to disk, captioned with the local VLM (`llama3.2-vision`), and embedded into vector search.
2. **Multimodal Hybrid Retrieval**: Images and text chunks compete on equal footing via 4-way parallel search (vector chunks, FTS chunks, vector image captions, FTS image captions) fused via Reciprocal Rank Fusion (RRF) and scored by the CrossEncoder reranker.
3. **Inline Image Citations**: When an image is relevant, the LLM cites it with `[doc_id:img_N]`. The API validates the citation and returns a structured `type: "image"` source object with `image_url` for inline rendering in chat UIs.
4. **Interactive Visual Q&A (`POST /images/{id}/ask`)**: Direct visual question-answering on a specific image, bypassing text retrieval.
5. **Conversational Image Auto-Routing**: When `conversation_id` is supplied, recently-shown images are tracked in Redis and follow-up visual questions are auto-routed to direct visual Q&A.
6. **Graceful Degradation**: The VLM model is optional. If `llama3.2-vision` is not pulled, captioning is skipped silently during ingestion without breaking document processing or text search.

---

## Quick Start

### 1. Prerequisites

- Docker ≥ 24 and Docker Compose v2
- Free disk space for local models
- No GPU required (CPU-only by default; GPU optional)

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env to set your secrets or customize model names
```

### 3. Start the stack

```bash
docker compose up -d --build
```

### 4. (Optional) Pull the Vision Model

To enable image captioning and visual Q&A:
```bash
docker exec rag-ollama-1 ollama pull llama3.2-vision
```
*Note: VLM captioning runs in the background arq worker. On CPU without GPU, captioning a document with diagrams may take a few minutes.*

### 5. Verify

```bash
curl http://localhost:8000/health
# → {"status":"ok","version":"3.0.0"}
```

- **API docs**: http://localhost:8000/docs
- **Supabase Studio**: http://localhost:3000

---

## API Reference

### Upload a document
```bash
curl -X POST http://localhost:8000/documents \
  -F "file=@sample_docs/architecture_diagram.pdf" \
  -F "tenant_id=00000000-0000-0000-0000-000000000001"
```

### Query the knowledge base (Multimodal RAG)
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the query latency for pgvector HNSW in the benchmark chart?",
    "tenant_id": "00000000-0000-0000-0000-000000000001",
    "conversation_id": "00000000-0000-0000-0000-000000000099"
  }'
```

**Response Example (with inline image citation):**
```json
{
  "query": "What is the query latency for pgvector HNSW in the benchmark chart?",
  "answer": "According to the benchmark chart [doc_uuid:img_0], pgvector HNSW achieves an average query latency of 12 ms.",
  "was_refused": false,
  "top_similarity_score": 1.0,
  "sources": [
    {
      "type": "image",
      "citation_id": "doc_uuid:img_0",
      "filename": "architecture_diagram.pdf",
      "page_number": 1,
      "score": 0.94,
      "image_url": "/images/files/00000000-0000-0000-0000-00000001/doc_uuid/page_1_img_0.png"
    }
  ],
  "latency_ms": 1850,
  "routed_to_image_qa": false
}
```

### Direct Visual Q&A on a specific image
```bash
curl -X POST http://localhost:8000/images/<image_id>/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the color of the pgvector bar in this chart?",
    "tenant_id": "00000000-0000-0000-0000-000000000001"
  }'
```

### Download / View Raw Image File
```bash
curl "http://localhost:8000/images/<image_id>/file?tenant_id=00000000-0000-0000-0000-000000000001"
```

---

## Smoke Tests

Generate sample test files (including PDF with chart) and run the end-to-end test suite:

```bash
# 1. Generate test PDF diagram
python3 smoke_tests/generate_test_pdf.py

# 2. Run test suite
python3 smoke_tests/run_smoke_tests.py \
  --tenant-id 00000000-0000-0000-0000-000000000001
```

---

## Architecture Diagram

```
Client
  │
  ├── POST /documents ──► Ingestion Worker (arq)
  │                         ├── Parse text & structure (pdfplumber)
  │                         ├── Chunk & Embed (sentence-transformers)
  │                         ├── Extract images (PyMuPDF)
  │                         ├── Caption images (local Ollama VLM)
  │                         └── Embed captions into images table (pgvector + GIN)
  │
  ├── POST /query     ──► Multimodal Hybrid Retrieval
  │                         ├── 4-way parallel search (vector/FTS on chunks + images)
  │                         ├── Reciprocal Rank Fusion (RRF)
  │                         ├── CrossEncoder reranking
  │                         ├── Confidence gate check
  │                         ├── Ollama Generation with [doc_id:img_N] citation rules
  │                         └── Return Answer + Sources (with image_url)
  │
  └── POST /images/{id}/ask ──► Direct VLM Visual Q&A
```
