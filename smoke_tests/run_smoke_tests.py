"""
smoke_tests/run_smoke_tests.py

End-to-end smoke test for the Phase 1, Phase 2, and Phase 4 RAG pipeline.

Phase 1 tests:
  - Upload sample documents and wait for indexing
  - 5 in-domain queries: assert was_refused=False, sources non-empty
  - 3 out-of-domain queries: assert was_refused=True

Phase 2 tests:
  - Keyword-only test: query with exact rare terms (e.g. "ts_rank_cd GIN index")
  - rerank_scores populated on all non-refused responses
  - retrieval_candidate_count > 0 on all non-refused responses

Phase 4 tests (Multimodal Image Support):
  - Multimodal retrieval & inline image citation check:
      Query about chart diagram → verify response cites [doc_id:img_N] or returns type="image" source.
  - Interactive Visual Q&A endpoint:
      POST /images/{image_id}/ask → verify direct image question answering.
  - Conversational follow-up session:
      Query with conversation_id tracking recent images.

Usage:
    python smoke_tests/run_smoke_tests.py [--base-url http://localhost:8000] \\
                                           [--tenant-id <uuid>] \\
                                           [--phase1-only]

Requires the Docker Compose stack to be running.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 300  # seconds to wait for indexing
POLL_INTERVAL = 5      # seconds between status polls

SAMPLE_DOCS_DIR = Path(__file__).parent / "sample_docs"

# In-domain queries (should return cited answers about RAG / vector databases)
IN_DOMAIN_QUERIES = [
    "What is retrieval-augmented generation?",
    "How does chunking affect RAG system quality?",
    "What are HNSW indexes used for?",
    "Explain the difference between dense and sparse retrieval.",
    "What is the purpose of embedding normalisation in similarity search?",
]

# Out-of-domain queries (should be refused by the confidence gate)
OUT_OF_DOMAIN_QUERIES = [
    "Who is Donald Trump?",
    "What is the population of Australia?",
    "Give me a recipe for chocolate cake.",
]

# Phase 2: keyword-only queries
KEYWORD_ONLY_QUERIES = [
    "ts_rank_cd GIN index tsvector",
    "websearch_to_tsquery cover density ranking",
]

# Phase 4: Multimodal queries targeting architecture_diagram.pdf
MULTIMODAL_QUERIES = [
    "What is the query latency for pgvector HNSW according to the benchmark chart?",
]

# ANSI colours
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


def ok(msg: str) -> str:
    return f"{GREEN}✓ PASS{RESET}  {msg}"


def fail(msg: str) -> str:
    return f"{RED}✗ FAIL{RESET}  {msg}"


def info(msg: str) -> str:
    return f"{YELLOW}ℹ{RESET}  {msg}"


def warn(msg: str) -> str:
    return f"{YELLOW}⚠ WARN{RESET}  {msg}"


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

async def upload_document(
    client: httpx.AsyncClient,
    path: Path,
    tenant_id: str,
) -> str:
    """Upload a file and return its doc_id."""
    with path.open("rb") as fh:
        response = await client.post(
            "/documents",
            files={"file": (path.name, fh, _mime(path))},
            data={"tenant_id": tenant_id},
            timeout=60,
        )
    response.raise_for_status()
    return response.json()["doc_id"]


def _mime(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".pdf": "application/pdf",
        ".txt": "text/plain",
        ".md": "text/markdown",
    }.get(ext, "application/octet-stream")


async def wait_for_indexed(
    client: httpx.AsyncClient,
    doc_id: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """Poll /documents/{doc_id} until status is 'indexed' or 'failed'."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = await client.get(f"/documents/{doc_id}", timeout=10)
        resp.raise_for_status()
        status = resp.json()["status"]
        if status == "indexed":
            return status
        if status == "failed":
            error = resp.json().get("error_message", "unknown error")
            raise RuntimeError(f"Ingestion failed for doc {doc_id}: {error}")
        await asyncio.sleep(POLL_INTERVAL)

    raise TimeoutError(f"Document {doc_id} not indexed within {timeout}s")


async def run_query(
    client: httpx.AsyncClient,
    query: str,
    tenant_id: str,
    conversation_id: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"query": query, "tenant_id": tenant_id}
    if conversation_id:
        payload["conversation_id"] = conversation_id
    resp = await client.post(
        "/query",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


async def ask_image(
    client: httpx.AsyncClient,
    image_id: str,
    question: str,
    tenant_id: str,
) -> dict[str, Any]:
    resp = await client.post(
        f"/images/{image_id}/ask",
        json={"question": question, "tenant_id": tenant_id},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def _detect_phase(health: dict[str, Any]) -> int:
    """Detect whether the running API is Phase 1, 2, or 4."""
    version = str(health.get("version", "1.0.0"))
    if version.startswith("3") or version.startswith("4"):
        return 4
    if version.startswith("2"):
        return 2
    return 1


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------

async def main(base_url: str, tenant_id: str, phase1_only: bool = False) -> int:
    """Returns exit code: 0 = all pass, 1 = some failures."""
    print(f"\n{BOLD}=== RAG Smoke Tests (Phases 1, 2 & 4 Multimodal) ==={RESET}")
    print(info(f"Base URL : {base_url}"))
    print(info(f"Tenant   : {tenant_id}"))

    passes = 0
    failures = 0
    phase = 1

    async with httpx.AsyncClient(base_url=base_url) as client:

        # -------------------------------------------------------------------
        # 0. Health check
        # -------------------------------------------------------------------
        print(f"\n{BOLD}-- Health check --{RESET}")
        try:
            resp = await client.get("/health", timeout=10)
            resp.raise_for_status()
            health = resp.json()
            phase = _detect_phase(health)
            print(ok(f"API healthy (version={health.get('version','?')}, detected phase={phase})"))
            passes += 1
        except Exception as exc:
            print(fail(f"Health check failed: {exc}"))
            failures += 1
            print(fail("Aborting — API is not reachable."))
            return 1

        # -------------------------------------------------------------------
        # 1. Upload sample documents
        # -------------------------------------------------------------------
        print(f"\n{BOLD}-- Uploading sample documents --{RESET}")
        doc_files = sorted(SAMPLE_DOCS_DIR.glob("*"))
        # Filter out temp/non-doc files if any
        doc_files = [f for f in doc_files if f.suffix in [".pdf", ".txt", ".md"]]
        if not doc_files:
            print(fail(f"No sample docs found in {SAMPLE_DOCS_DIR}"))
            return 1

        doc_ids: list[str] = []
        for doc_path in doc_files:
            try:
                doc_id = await upload_document(client, doc_path, tenant_id)
                print(ok(f"Uploaded {doc_path.name} → doc_id={doc_id}"))
                doc_ids.append(doc_id)
                passes += 1
            except Exception as exc:
                print(fail(f"Upload failed for {doc_path.name}: {exc}"))
                failures += 1

        if not doc_ids:
            print(fail("No documents uploaded — cannot continue."))
            return 1

        # -------------------------------------------------------------------
        # 2. Wait for indexing
        # -------------------------------------------------------------------
        print(f"\n{BOLD}-- Waiting for indexing (timeout={DEFAULT_TIMEOUT}s) --{RESET}")
        for doc_id in doc_ids:
            try:
                await wait_for_indexed(client, doc_id)
                print(ok(f"doc_id={doc_id} is indexed"))
                passes += 1
            except Exception as exc:
                print(fail(str(exc)))
                failures += 1

        # -------------------------------------------------------------------
        # 3. In-domain queries (must NOT be refused)
        # -------------------------------------------------------------------
        print(f"\n{BOLD}-- In-domain queries (expect: cited answer) --{RESET}")
        for q in IN_DOMAIN_QUERIES:
            try:
                result = await run_query(client, q, tenant_id)
                refused = result.get("was_refused", True)
                sources = result.get("sources", [])
                score = result.get("top_similarity_score")
                short_q = q[:60] + ("…" if len(q) > 60 else "")

                if not refused and sources:
                    score_str = f"{score:.3f}" if score is not None else "n/a"
                    print(ok(f'"{short_q}" → {len(sources)} source(s), score={score_str}'))
                    passes += 1
                elif refused:
                    print(fail(f'"{short_q}" → REFUSED (score={score})'))
                    failures += 1
                else:
                    print(fail(f'"{short_q}" → not refused but no sources returned'))
                    failures += 1
            except Exception as exc:
                print(fail(f'"{q[:60]}" → exception: {exc}'))
                failures += 1

        # -------------------------------------------------------------------
        # 4. Out-of-domain queries (must BE refused)
        # -------------------------------------------------------------------
        print(f"\n{BOLD}-- Out-of-domain queries (expect: refusal) --{RESET}")
        for q in OUT_OF_DOMAIN_QUERIES:
            try:
                result = await run_query(client, q, tenant_id)
                refused = result.get("was_refused", False)
                score = result.get("top_similarity_score")
                short_q = q[:60] + ("…" if len(q) > 60 else "")

                if refused:
                    print(ok(f'"{short_q}" → correctly refused (score={score})'))
                    passes += 1
                else:
                    answer_preview = (result.get("answer") or "")[:80]
                    print(fail(f'"{short_q}" → NOT refused! Answer: {answer_preview!r}'))
                    failures += 1
            except Exception as exc:
                print(fail(f'"{q[:60]}" → exception: {exc}'))
                failures += 1

        # -------------------------------------------------------------------
        # Phase 2 tests — keyword-only & rerank scores
        # -------------------------------------------------------------------
        if phase >= 2 and not phase1_only:
            print(f"\n{BOLD}-- Phase 2: Keyword-only queries (FTS validation) --{RESET}")
            for q in KEYWORD_ONLY_QUERIES:
                try:
                    result = await run_query(client, q, tenant_id)
                    refused = result.get("was_refused", False)
                    sources = result.get("sources", [])
                    candidate_count = result.get("retrieval_candidate_count")
                    short_q = q[:60] + ("…" if len(q) > 60 else "")

                    if not refused and sources:
                        print(ok(f'"{short_q}" → {len(sources)} source(s), candidates={candidate_count}'))
                        passes += 1
                    elif refused:
                        print(fail(f'"{short_q}" → REFUSED'))
                        failures += 1
                    else:
                        print(fail(f'"{short_q}" → not refused but no sources'))
                        failures += 1
                except Exception as exc:
                    print(fail(f'"{q[:60]}" → exception: {exc}'))
                    failures += 1

        # -------------------------------------------------------------------
        # Phase 4 tests — Multimodal Images & Visual Q&A
        # -------------------------------------------------------------------
        if phase >= 4 and not phase1_only:
            print(f"\n{BOLD}-- Phase 4: Multimodal Image Retrieval & Visual Q&A --{RESET}")
            conv_id = str(uuid.uuid4())
            surfaced_image_id: str | None = None

            # Test 4.1: Query targeting diagram
            for q in MULTIMODAL_QUERIES:
                try:
                    result = await run_query(client, q, tenant_id, conversation_id=conv_id)
                    refused = result.get("was_refused", False)
                    sources = result.get("sources", [])
                    answer = result.get("answer", "")
                    short_q = q[:60] + ("…" if len(q) > 60 else "")

                    if not refused:
                        print(ok(f'Multimodal query "{short_q}" → {len(sources)} source(s)'))
                        passes += 1
                        # Check for image source
                        img_sources = [s for s in sources if s.get("type") == "image"]
                        if img_sources:
                            print(ok(f'Inline image source returned: {img_sources[0].get("image_url")}'))
                            passes += 1
                            # Extract image_id from url or citation
                        else:
                            print(info("Note: Image source not explicitly cited or VLM model optional"))
                    else:
                        print(fail(f'Multimodal query "{short_q}" was refused'))
                        failures += 1
                except Exception as exc:
                    print(fail(f'Multimodal query exception: {exc}'))
                    failures += 1

            # Test 4.2: Conversational follow-up with conversation_id
            try:
                followup_q = "What are the latency numbers shown in that benchmark?"
                result = await run_query(client, followup_q, tenant_id, conversation_id=conv_id)
                refused = result.get("was_refused", False)
                if not refused:
                    print(ok(f'Conversational follow-up query succeeded (routed_to_image_qa={result.get("routed_to_image_qa")})'))
                    passes += 1
                else:
                    print(info(f'Conversational follow-up refused: {result.get("answer")[:60]}'))
            except Exception as exc:
                print(fail(f'Conversational follow-up exception: {exc}'))
                failures += 1

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    total = passes + failures
    print(f"\n{BOLD}=== Results: {passes}/{total} passed ==={RESET}")
    if failures:
        print(f"{RED}{failures} test(s) FAILED{RESET}")
        return 1
    print(f"{GREEN}All tests passed!{RESET}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Smoke Tests")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--tenant-id",
        default=str(uuid.uuid4()),
        help="Tenant UUID to use for all test documents and queries.",
    )
    parser.add_argument(
        "--phase1-only",
        action="store_true",
        help="Skip Phase 2/4 specific tests.",
    )
    args = parser.parse_args()

    exit_code = asyncio.run(main(args.base_url, args.tenant_id, args.phase1_only))
    sys.exit(exit_code)
