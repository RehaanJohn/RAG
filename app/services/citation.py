"""
app/services/citation.py

Citation parsing, validation, and resolution.

Citation format used in LLM output:
  Text chunks: [doc_uuid:chunk_index]         e.g. [3f2504e0...:4]
  Images:      [doc_uuid:img_image_index]      e.g. [3f2504e0...:img_2]

Three validation rules are enforced (in order):
  1. Every sentence containing a factual claim must carry ≥1 citation.
  2. Every cited ID must exist in the retrieved set (no hallucinated citations).
  3. Every cited ID must resolve to a real chunk or image in our DB results.
"""
from __future__ import annotations

import logging
import re
import uuid

from app.models.schemas import ChunkResult, DocumentResponse, ImageResult, SourceObject

logger = logging.getLogger(__name__)

# Matches text citations:  [uuid:N]      e.g. [3f2504e0...:4]
# Matches image citations: [uuid:img_N]  e.g. [3f2504e0...:img_2]
_CITATION_RE = re.compile(
    r"\[([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}:(?:img_)?\d+)\]",
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
    images: list[ImageResult] | None = None,
    base_url: str = "",
) -> list[SourceObject]:
    """
    Convert raw citation IDs to structured SourceObject instances.

    Handles both text chunk citations (uuid:N) and image citations (uuid:img_N).
    Image entries are enriched with image_url so clients can render them inline.

    Args:
        cited_ids: IDs parsed from the model output.
        chunks: Retrieved text chunk results.
        documents: Map from doc_id → DocumentResponse.
        images: Retrieved image results (Phase 4).
        base_url: Optional base URL prefix for image URLs.

    Returns:
        Deduplicated list of SourceObject, ordered by first appearance.
    """
    from app.services.image_store import storage_path_to_url

    # Build lookups
    chunk_map: dict[str, ChunkResult] = {c.citation_id: c for c in chunks}
    image_map: dict[str, ImageResult] = {}
    if images:
        image_map = {img.citation_id: img for img in images}

    seen: set[str] = set()
    sources: list[SourceObject] = []

    for cid in cited_ids:
        if cid in seen:
            continue
        seen.add(cid)

        # --- Text chunk citation ---
        if cid in chunk_map:
            chunk = chunk_map[cid]
            doc = documents.get(chunk.doc_id)
            filename = doc.filename if doc else "unknown"
            sources.append(
                SourceObject(
                    type="text",
                    citation_id=cid,
                    filename=filename,
                    page_number=chunk.metadata.get("page_number"),
                    section_path=chunk.metadata.get("section_path"),
                    score=chunk.score,
                )
            )
            continue

        # --- Image citation ---
        if cid in image_map:
            img = image_map[cid]
            doc = documents.get(img.doc_id)
            filename = doc.filename if doc else "unknown"
            image_url = storage_path_to_url(img.storage_path, base_url)
            sources.append(
                SourceObject(
                    type="image",
                    citation_id=cid,
                    filename=filename,
                    page_number=img.page_number,
                    score=img.score,
                    image_url=image_url,
                )
            )
            continue

        # --- Citation not found in retrieved set ---
        logger.debug("Citation %s not in retrieved set — skipping.", cid)

    return sources
