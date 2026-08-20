"""
app/services/citation.py

Citation parsing, validation, and resolution.

Citation format used in LLM output: [doc_uuid:chunk_index]
Example: [3f2504e0-4f89-11d3-9a0c-0305e82c3301:4]

Three validation rules are enforced (in order):
  1. Every sentence containing a factual claim must carry ≥1 citation.
  2. Every cited ID must exist in the retrieved set (no hallucinated citations).
  3. Every cited ID must resolve to a real chunk in our DB results.
"""
from __future__ import annotations

import logging
import re
import uuid

from app.models.schemas import ChunkResult, DocumentResponse, SourceObject

logger = logging.getLogger(__name__)

# Matches: [<uuid>:<integer>]
# UUID is hex with optional hyphens; chunk_index is a non-negative integer.
_CITATION_RE = re.compile(
    r"\[([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}:\d+)\]",
    re.IGNORECASE,
)

# Sentences that look factual: contain a verb phrase.
# This is a heuristic — we split on sentence boundaries and check for citations.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Known non-factual sentence starters (questions, conjunctions, etc.)
_NON_FACTUAL_STARTERS = frozenset(
    ["however,", "for example,", "in addition,", "note:", "note that"]
)


def parse_citations(text: str) -> list[str]:
    """
    Extract all citation tags from a generated answer.

    Returns:
        List of raw citation strings, e.g. ["uuid:4", "uuid:7"].
    """
    return _CITATION_RE.findall(text)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, preserving refusal boilerplate."""
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def check_uncited_claims(text: str) -> tuple[bool, list[str]]:
    """
    Check whether every sentence that appears to contain a factual claim
    has at least one inline citation.

    Returns:
        (ok, uncited_sentences)
        ok=True means all detectable factual sentences have citations.
    """
    sentences = _split_sentences(text)
    uncited: list[str] = []

    for sentence in sentences:
        lower = sentence.lower()

        # Skip short fragments (questions, headings, connectors)
        if len(sentence) < 20:
            continue
        if any(lower.startswith(s) for s in _NON_FACTUAL_STARTERS):
            continue
        # Skip the fixed refusal message
        if "couldn't find anything" in lower:
            continue

        # If this sentence contains no citation tag, flag it
        if not _CITATION_RE.search(sentence):
            uncited.append(sentence)

    ok = len(uncited) == 0
    if not ok:
        logger.warning("Found %d uncited sentence(s).", len(uncited))
    return ok, uncited


def validate_citations(
    cited_ids: list[str],
    retrieved_set: set[str],
) -> tuple[bool, list[str]]:
    """
    Ensure every citation ID in the model output was actually retrieved.
    Any citation not in retrieved_set is a hallucinated citation.

    Returns:
        (ok, hallucinated_ids)
    """
    hallucinated = [cid for cid in cited_ids if cid not in retrieved_set]
    ok = len(hallucinated) == 0
    if not ok:
        logger.warning("Hallucinated citation IDs: %s", hallucinated)
    return ok, hallucinated


def resolve_citations(
    cited_ids: list[str],
    chunks: list[ChunkResult],
    documents: dict[uuid.UUID, DocumentResponse],
) -> list[SourceObject]:
    """
    Convert raw citation IDs to structured SourceObject instances that the
    client can use to link back to the original document.

    Args:
        cited_ids: IDs parsed from the model output, e.g. ["uuid:3"].
        chunks: The retrieved chunk results for this query.
        documents: Map from doc_id → DocumentResponse.

    Returns:
        Deduplicated list of SourceObject, ordered by first appearance.
    """
    # Build lookup: citation_id → ChunkResult
    chunk_map: dict[str, ChunkResult] = {c.citation_id: c for c in chunks}

    seen: set[str] = set()
    sources: list[SourceObject] = []

    for cid in cited_ids:
        if cid in seen:
            continue
        seen.add(cid)

        chunk = chunk_map.get(cid)
        if chunk is None:
            # Safety: skip IDs that don't map to a retrieved chunk
            logger.debug("Citation %s not in retrieved chunk map — skipping.", cid)
            continue

        doc = documents.get(chunk.doc_id)
        filename = doc.filename if doc else "unknown"

        sources.append(
            SourceObject(
                citation_id=cid,
                filename=filename,
                page_number=chunk.metadata.get("page_number"),
                section_path=chunk.metadata.get("section_path"),
                score=chunk.score,
            )
        )

    return sources
