"""
app/services/generation.py

Assembles the LLM prompt from retrieved chunks and images, calls local Ollama,
and validates citations post-generation.

All calls are to the local Ollama endpoint — no external LLM APIs.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Sequence, Union

import httpx

from app.config import settings
from app.models.schemas import (
    ChunkResult,
    DocumentResponse,
    ImageResult,
    QueryResponse,
    SourceObject,
)
from app.services.citation import (
    check_uncited_claims,
    parse_citations,
    resolve_citations,
    validate_citations,
)

logger = logging.getLogger(__name__)

# Fixed refusal message — must appear verbatim in all refusal responses.
REFUSAL_MESSAGE = (
    "I couldn't find anything in the knowledge base that answers this. "
    "Try rephrasing, or this may be outside what's currently indexed."
)

Candidate = Union[ChunkResult, ImageResult]

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are a precise research assistant. Your ONLY job is to answer questions
based on the document excerpts and images provided below. Follow these rules WITHOUT EXCEPTION:

1. Answer ONLY using information from the provided excerpts and images.
2. Do NOT use outside knowledge, general knowledge, or any information not
   present in the excerpts — even if you believe it to be true.
3. For EVERY factual claim you make, cite the source chunk or image inline using the
   exact citation format: [doc_id:chunk_index] for text or [doc_id:img_index] for images.
   Never write a factual sentence without an inline citation.
4. When an image is relevant to your answer, cite it using [doc_id:img_index] so the
   user can view the source image.
5. If the excerpts and images do not contain enough information to answer the question,
   output EXACTLY the following refusal message and NOTHING else:
   "{refusal}"
6. Do not hallucinate, invent, or extrapolate beyond what the excerpts state.

--- DOCUMENT EXCERPTS & IMAGES ---
{context}
--- END EXCERPTS ---"""

_SYSTEM_PROMPT_STRICT = """\
You are a precise research assistant. Your ONLY job is to answer questions
based on the document excerpts and images provided below. CRITICAL REMINDER:

⚠ EVERY SINGLE factual sentence MUST end with a citation tag [doc_id:chunk_index] or [doc_id:img_index].
⚠ Do NOT use any knowledge outside the provided excerpts.
⚠ If the answer isn't in the excerpts, respond ONLY with the exact refusal message.

CITATION FORMAT: [doc_id:chunk_index] for text, [doc_id:img_index] for images — use the exact IDs shown in the excerpts.
If you omit even one citation, your entire response will be discarded.

--- DOCUMENT EXCERPTS & IMAGES ---
{context}
--- END EXCERPTS ---"""

_USER_PROMPT_TEMPLATE = "Question: {query}"


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------


def _build_context_block(items: Sequence[Candidate]) -> str:
    """
    Format retrieved chunks and images into a numbered context block for the LLM.
    Each excerpt is tagged with a stable citation ID.
    """
    lines: list[str] = []
    for item in items:
        cid = item.citation_id
        if isinstance(item, ImageResult):
            meta_parts = []
            if item.page_number:
                meta_parts.append(f"page {item.page_number}")
            meta_parts.append("image")
            meta_str = f" ({', '.join(meta_parts)})"
            caption_text = item.caption.strip() if item.caption else "[Image present without caption]"
            lines.append(f"[{cid}]{meta_str}:\n{caption_text}")
        else:
            chunk: ChunkResult = item
            meta_parts = []
            if pn := chunk.metadata.get("page_number"):
                meta_parts.append(f"page {pn}")
            if sp := chunk.metadata.get("section_path"):
                meta_parts.append(f'section "{sp}"')
            meta_str = f" ({', '.join(meta_parts)})" if meta_parts else ""
            lines.append(f"[{cid}]{meta_str}:\n{chunk.content.strip()}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Ollama call (synchronous-over-async via httpx)
# ---------------------------------------------------------------------------


async def _call_ollama(system_prompt: str, user_prompt: str) -> str:
    """
    Call the local Ollama /api/chat endpoint and return the full response text.
    stream=False so we get a single JSON response.
    """
    payload = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,   # Low temp for factual, citation-focused answers
            "num_predict": 1024,
        },
    }

    async with httpx.AsyncClient(timeout=settings.ollama_timeout) as client:
        response = await client.post(
            f"{settings.ollama_base_url}/api/chat",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

    # Ollama /api/chat returns: { "message": { "role": "assistant", "content": "..." } }
    return data["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Generation with citation validation
# ---------------------------------------------------------------------------


async def generate_answer(
    query: str,
    chunks: Sequence[ChunkResult],
    documents: dict[uuid.UUID, DocumentResponse],
    images: Sequence[ImageResult] | None = None,
    base_url: str = "",
) -> tuple[str, list[SourceObject], bool]:
    """
    Generate an answer from retrieved chunks & images using local Ollama.
    Validates citations post-generation, retrying once with a stricter prompt
    if validation fails. Falls back to refusal on second failure.

    Returns:
        (answer_text, sources, was_refused)
    """
    all_items: list[Candidate] = list(chunks) + (list(images) if images else [])
    context_block = _build_context_block(all_items)
    retrieved_ids: set[str] = {c.citation_id for c in all_items}
    user_prompt = _USER_PROMPT_TEMPLATE.format(query=query)

    for attempt in range(2):
        if attempt == 0:
            system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
                refusal=REFUSAL_MESSAGE,
                context=context_block,
            )
        else:
            # Stricter reminder on retry
            system_prompt = _SYSTEM_PROMPT_STRICT.format(context=context_block)
            logger.info("Retrying generation with stricter citation prompt.")

        try:
            raw_answer = await _call_ollama(system_prompt, user_prompt)
        except Exception as exc:
            logger.error("Ollama call failed on attempt %d: %s", attempt + 1, exc)
            break

        # --- Check if the model self-refused ---
        if REFUSAL_MESSAGE.lower() in raw_answer.lower():
            logger.info("Model self-refused on attempt %d.", attempt + 1)
            return REFUSAL_MESSAGE, [], True

        # --- Citation validation (step 1): uncited factual claims ---
        if settings.citation_retry_enabled:
            claims_ok, _ = check_uncited_claims(raw_answer)
            if not claims_ok and attempt == 0:
                logger.warning("Uncited claims found; retrying.")
                continue

            # --- Citation validation (step 2): hallucinated IDs ---
            cited_ids = parse_citations(raw_answer)
            hall_ok, _ = validate_citations(cited_ids, retrieved_ids)
            if not hall_ok and attempt == 0:
                logger.warning("Hallucinated citations found; retrying.")
                continue

            if attempt == 1:
                # Second attempt: accept whatever we have if structurally OK
                cited_ids = parse_citations(raw_answer)
                hall_ok_2, _ = validate_citations(cited_ids, retrieved_ids)
                if not hall_ok_2:
                    logger.error("Citation validation failed on retry — falling back to refusal.")
                    return REFUSAL_MESSAGE, [], True
        else:
            cited_ids = parse_citations(raw_answer)

        # --- All checks passed: resolve citations ---
        cited_ids = parse_citations(raw_answer)
        sources = resolve_citations(
            cited_ids,
            list(chunks),
            documents,
            images=list(images) if images else None,
            base_url=base_url,
        )

        logger.info(
            "Generation succeeded (attempt=%d, citations=%d, sources=%d).",
            attempt + 1,
            len(cited_ids),
            len(sources),
        )
        return raw_answer, sources, False

    # Both attempts failed or Ollama error
    logger.error("Falling back to refusal after %d attempt(s).", 2)
    return REFUSAL_MESSAGE, [], True
