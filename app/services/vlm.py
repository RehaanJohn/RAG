"""
app/services/vlm.py

Local Vision-Language Model service via Ollama.

IMPORTANT — VLM IS OPTIONAL:
  If the configured VLM_MODEL is not pulled into the Ollama instance,
  all functions degrade gracefully:
    - caption_image()  → returns None  (no exception)
    - ask_image()      → returns empty string with vlm_unavailable=True
    - is_available()   → returns False

This means image captioning is silently skipped during ingestion, and the
/images/{id}/ask endpoint returns a structured "VLM not available" response
rather than a 500. The rest of the pipeline (text search, chat) is unaffected.

Pull the VLM model at any time with:
    docker exec rag-ollama-1 ollama pull llama3.2-vision
After pulling, restart the app container for captioning to take effect.
"""
from __future__ import annotations

import base64
import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixed captioning prompt
# ---------------------------------------------------------------------------

_CAPTION_PROMPT = (
    "Describe this image in detail, including any text, labels, measurements, "
    "data values, chart axes, diagram components, or annotations visible in it. "
    "Be specific and complete — your description will be used to index this image "
    "for search. Do not speculate beyond what is visually present."
)

# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

_vlm_available: bool | None = None  # cached after first check


async def is_available(force_recheck: bool = False) -> bool:
    """
    Check whether the configured VLM model is loaded in Ollama.
    Result is cached after the first successful check.
    """
    global _vlm_available

    if _vlm_available is True and not force_recheck:
        return True

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            # Match on model name prefix (e.g. "llama3.2-vision" matches "llama3.2-vision:latest")
            model_prefix = settings.vlm_model.split(":")[0]
            _vlm_available = any(m.startswith(model_prefix) for m in models)
            if not _vlm_available:
                logger.info(
                    "VLM model '%s' not found in Ollama. "
                    "Image captioning is disabled. "
                    "Pull with: docker exec rag-ollama-1 ollama pull %s",
                    settings.vlm_model, settings.vlm_model,
                )
    except Exception as exc:
        logger.debug("VLM availability check failed: %s", exc)
        _vlm_available = False

    return _vlm_available  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Caption generation
# ---------------------------------------------------------------------------

async def caption_image(image_bytes: bytes) -> str | None:
    """
    Send an image to the local VLM and return a detailed caption.

    Returns:
        Caption string on success.
        None if VLM is not available or captioning fails — caller should
        log and skip, not raise.
    """
    if not await is_available():
        return None

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "model": settings.vlm_model,
        "messages": [
            {
                "role": "user",
                "content": _CAPTION_PROMPT,
                "images": [b64],
            }
        ],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 512},
    }

    try:
        async with httpx.AsyncClient(timeout=settings.vlm_timeout) as client:
            resp = await client.post(
                f"{settings.ollama_base_url}/api/chat",
                json=payload,
            )
            resp.raise_for_status()
            caption = resp.json()["message"]["content"].strip()
            logger.debug("Caption generated (%d chars).", len(caption))
            return caption
    except Exception as exc:
        logger.warning("VLM captioning failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Direct visual Q&A
# ---------------------------------------------------------------------------

async def ask_image(image_bytes: bytes, question: str) -> tuple[str, bool]:
    """
    Send an image + question directly to the VLM for visual Q&A.
    This bypasses retrieval entirely — the image is the sole context.

    Returns:
        (answer, vlm_unavailable)
        vlm_unavailable=True means the model is not loaded; answer will be "".
    """
    if not await is_available():
        return "", True

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "model": settings.vlm_model,
        "messages": [
            {
                "role": "user",
                "content": question,
                "images": [b64],
            }
        ],
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 1024},
    }

    try:
        async with httpx.AsyncClient(timeout=settings.vlm_timeout) as client:
            resp = await client.post(
                f"{settings.ollama_base_url}/api/chat",
                json=payload,
            )
            resp.raise_for_status()
            answer = resp.json()["message"]["content"].strip()
            return answer, False
    except Exception as exc:
        logger.error("VLM ask_image failed: %s", exc)
        return "", False


# ---------------------------------------------------------------------------
# Image-reference classifier (lightweight)
# ---------------------------------------------------------------------------

async def classify_image_reference(
    question: str,
    recent_captions: list[str],
) -> tuple[bool, int | None]:
    """
    Ask the text LLM whether this question refers to one of the recently-shown images.

    Only called when conversation_id is present and recent images exist in Redis.
    Uses the text model (OLLAMA_MODEL), not the VLM — it's a fast text classification.

    Returns:
        (is_image_reference: bool, index_in_recent_list: int | None)
        index_in_recent_list is 0-based; 0 = most recently shown image.
    """
    if not recent_captions:
        return False, None

    captions_block = "\n".join(
        f"[Image {i}]: {cap[:200]}" for i, cap in enumerate(recent_captions)
    )

    prompt = (
        f"The user just asked: \"{question}\"\n\n"
        f"In this conversation, these images were recently shown:\n{captions_block}\n\n"
        "Does the user's question appear to be specifically about one of these images "
        "(e.g. asking about a detail in the diagram, referencing 'that chart', "
        "'the second image', 'what does this graph show', etc.)?\n\n"
        "Reply with EXACTLY one of:\n"
        "  NO\n"
        "  YES:0\n"
        "  YES:1\n"
        "  YES:2\n"
        "(where the number is the 0-based index of the image being referenced)\n"
        "Reply with only the above token, nothing else."
    )

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{settings.ollama_base_url}/api/chat",
                json={
                    "model": settings.ollama_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 8},
                },
            )
            resp.raise_for_status()
            raw = resp.json()["message"]["content"].strip().upper()

        if raw == "NO" or not raw.startswith("YES"):
            return False, None
        parts = raw.split(":")
        idx = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 0
        idx = min(idx, len(recent_captions) - 1)
        logger.info("Image-reference classifier: YES (index=%d)", idx)
        return True, idx

    except Exception as exc:
        logger.debug("Image-reference classification failed: %s", exc)
        return False, None
