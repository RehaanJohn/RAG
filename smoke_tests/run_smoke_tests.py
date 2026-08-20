"""
smoke_tests/run_smoke_tests.py

End-to-end smoke test for the Phase 1 + Phase 2 RAG pipeline.

Phase 1 tests:
  - Upload sample documents and wait for indexing
  - 5 in-domain queries: assert was_refused=False, sources non-empty
  - 3 out-of-domain queries: assert was_refused=True

Phase 2 tests (additional):
  - Keyword-only test: query with exact rare terms (e.g. "ts_rank_cd GIN index")
    → assert was_refused=False (FTS should find it even if vector search misses)
  - rerank_scores populated on all non-refused responses
  - retrieval_candidate_count > 0 on all non-refused responses
  - Re-run OOD refusals to confirm gate still works under new reranker-based gate

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

# Phase 2: keyword-only queries — rare exact terms that only appear in
# chunking_guide.txt and are unlikely to be retrieved by vector search alone.
# If hybrid retrieval is working, FTS must find these.
KEYWORD_ONLY_QUERIES = [
    "ts_rank_cd GIN index tsvector",
    "websearch_to_tsquery cover density ranking",
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
) -> dict[str, Any]:
    resp = await client.post(
        "/query",
        json={"query": query, "tenant_id": tenant_id},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def _detect_phase(health: dict[str, Any]) -> int:
    """Detect whether the running API is Phase 1 or Phase 2."""
    version = health.get("version", "1.0.0")
    return 2 if version.startswith("2") else 1


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------

async def main(base_url: str, tenant_id: str, phase1_only: bool = False) -> int:
    """Returns exit code: 0 = all pass, 1 = some failures."""
    print(f"\n{BOLD}=== RAG Smoke Tests (Phase 1 + Phase 2) ==={RESET}")
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
            print(ok(f"API healthy (version={health.get('version','?')}, phase={phase})"))
            passes += 1
        except Exception as exc:
            print(fail(f"Health check failed: {exc}"))
            failures += 1
            print(fail("Aborting — API is not reachable."))
            return 1

        if phase1_only:
            print(info("--phase1-only flag set: skipping Phase 2 tests"))

        # -------------------------------------------------------------------
        # 1. Upload sample documents
        # -------------------------------------------------------------------
        print(f"\n{BOLD}-- Uploading sample documents --{RESET}")
        doc_files = sorted(SAMPLE_DOCS_DIR.glob("*"))
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
        # Phase 2 tests — only run when connected to a Phase 2 API
        # -------------------------------------------------------------------
        if phase >= 2 and not phase1_only:

            # -----------------------------------------------------------------
            # 5. Keyword-only queries (Phase 2: FTS must find rare exact terms)
            # -----------------------------------------------------------------
            print(f"\n{BOLD}-- Phase 2: Keyword-only queries (FTS validation) --{RESET}")
            print(info("These queries contain exact rare terms only in chunking_guide.txt"))
            print(info("They verify FTS retrieves content vector search alone might miss"))

            for q in KEYWORD_ONLY_QUERIES:
                try:
                    result = await run_query(client, q, tenant_id)
                    refused = result.get("was_refused", False)
                    sources = result.get("sources", [])
                    candidate_count = result.get("retrieval_candidate_count")
                    short_q = q[:60] + ("…" if len(q) > 60 else "")

                    if not refused and sources:
                        print(ok(
                            f'"{short_q}" → {len(sources)} source(s), '
                            f'candidates={candidate_count}'
                        ))
                        passes += 1
                    elif refused:
                        print(fail(
                            f'"{short_q}" → REFUSED '
                            f'(FTS may not have found the keyword doc — '
                            f'check chunking_guide.txt was indexed)'
                        ))
                        failures += 1
                    else:
                        print(fail(f'"{short_q}" → not refused but no sources'))
                        failures += 1
                except Exception as exc:
                    print(fail(f'"{q[:60]}" → exception: {exc}'))
                    failures += 1

            # -----------------------------------------------------------------
            # 6. Rerank scores populated on non-refused responses
            # -----------------------------------------------------------------
            print(f"\n{BOLD}-- Phase 2: Rerank scores presence check --{RESET}")
            all_queries = IN_DOMAIN_QUERIES + KEYWORD_ONLY_QUERIES
            rerank_check_count = 0
            for q in all_queries[:3]:  # spot-check first 3 in-domain
                try:
                    result = await run_query(client, q, tenant_id)
                    refused = result.get("was_refused", False)
                    rerank_scores = result.get("rerank_scores")
                    candidate_count = result.get("retrieval_candidate_count")
                    short_q = q[:55] + ("…" if len(q) > 55 else "")

                    if refused:
                        print(info(f'"{short_q}" was refused — skipping rerank check'))
                        continue

                    # Non-refused queries must have rerank_scores populated
                    if rerank_scores and len(rerank_scores) > 0:
                        print(ok(
                            f'"{short_q}" → rerank_scores present '
                            f'({len(rerank_scores)} chunks), candidates={candidate_count}'
                        ))
                        passes += 1
                        rerank_check_count += 1
                    else:
                        print(fail(
                            f'"{short_q}" → rerank_scores missing or empty '
                            f'(got: {rerank_scores!r})'
                        ))
                        failures += 1
                except Exception as exc:
                    print(fail(f'"{q[:55]}" → exception: {exc}'))
                    failures += 1

            if rerank_check_count == 0:
                print(warn(
                    "Could not verify rerank_scores — all spot-check queries were refused. "
                    "Try lowering GATE_RERANK_THRESHOLD or uploading more documents."
                ))

            # -----------------------------------------------------------------
            # 7. retrieval_candidate_count > 0 for non-refused queries
            # -----------------------------------------------------------------
            print(f"\n{BOLD}-- Phase 2: Candidate count > 0 check --{RESET}")
            try:
                result = await run_query(client, IN_DOMAIN_QUERIES[0], tenant_id)
                if not result.get("was_refused"):
                    count = result.get("retrieval_candidate_count")
                    if count and count > 0:
                        print(ok(f"retrieval_candidate_count={count} (> 0)"))
                        passes += 1
                    else:
                        print(fail(f"retrieval_candidate_count={count!r} (expected > 0)"))
                        failures += 1
                else:
                    print(info("Query was refused — skipping candidate count check"))
            except Exception as exc:
                print(fail(f"Candidate count check failed: {exc}"))
                failures += 1

        elif not phase1_only:
            print(f"\n{YELLOW}Skipping Phase 2 tests (API is Phase {phase}){RESET}")

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
    parser = argparse.ArgumentParser(description="RAG Phase 1 + Phase 2 smoke tests")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--tenant-id",
        default=str(uuid.uuid4()),
        help="Tenant UUID to use for all test documents and queries.",
    )
    parser.add_argument(
        "--phase1-only",
        action="store_true",
        help="Skip Phase 2 specific tests even if the API is Phase 2.",
    )
    args = parser.parse_args()

    exit_code = asyncio.run(main(args.base_url, args.tenant_id, args.phase1_only))
    sys.exit(exit_code)
