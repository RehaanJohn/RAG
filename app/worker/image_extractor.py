"""
app/worker/image_extractor.py

Extract embedded raster images from PDF files using PyMuPDF (fitz).

PyMuPDF is the primary extractor. For pages where no embedded raster images
are found but the page contains significant vector graphics (diagrams drawn
purely in PDF vector commands), we fall back to full-page rendering at 150 DPI.
This ensures vector-drawn charts and diagrams are captured.

Filtering:
  - Images smaller than IMAGE_MIN_SIZE_PX in either dimension are discarded
    (icons, decorators, horizontal rules, etc.)
  - Maximum IMAGE_MAX_PER_PAGE images per page to guard against tiled patterns.
  - Duplicate images within a document (same xref) are deduplicated.

All functions return plain Python objects — no DB or HTTP calls here.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ExtractedImage:
    """A single image extracted from a PDF page."""
    page_number: int        # 1-based
    image_index: int        # 0-based within the whole document (not per-page)
    image_bytes: bytes      # PNG-encoded
    width_px: int
    height_px: int
    bbox: dict[str, float] | None = None  # {x0, y0, x1, y1} in page points
    xref: int | None = None               # PyMuPDF internal ref for dedup


def extract_images_from_pdf(path: Path) -> list[ExtractedImage]:
    """
    Extract images from a PDF. Returns a flat list of ExtractedImage ordered
    by (page_number, image_index).

    Graceful degradation:
      - If PyMuPDF is not installed, logs a warning and returns [].
      - If a single page fails extraction, logs and continues.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.warning(
            "PyMuPDF (fitz) not installed — image extraction disabled. "
            "Add 'PyMuPDF>=1.24.0' to requirements.txt."
        )
        return []

    results: list[ExtractedImage] = []
    seen_xrefs: set[int] = set()
    global_image_index = 0
    min_px = settings.image_min_size_px
    max_per_page = settings.image_max_per_page

    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        logger.error("Failed to open PDF with PyMuPDF: %s — %s", path, exc)
        return []

    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_number = page_num + 1  # 1-based
            page_images: list[ExtractedImage] = []

            try:
                # --- Embedded raster images ---
                image_list = page.get_images(full=True)
                for img_info in image_list:
                    if len(page_images) >= max_per_page:
                        break

                    xref = img_info[0]
                    if xref in seen_xrefs:
                        continue

                    try:
                        base_image = doc.extract_image(xref)
                        img_bytes_raw = base_image["image"]
                        w = base_image.get("width", 0)
                        h = base_image.get("height", 0)

                        if w < min_px or h < min_px:
                            logger.debug(
                                "Skipping small image %dx%d on page %d",
                                w, h, page_number,
                            )
                            continue

                        # Convert to PNG for uniform output
                        png_bytes = _to_png(img_bytes_raw, base_image.get("ext", "png"))
                        if not png_bytes:
                            continue

                        # Get bounding box from image placement on page
                        bbox = _get_image_bbox(page, xref)

                        page_images.append(ExtractedImage(
                            page_number=page_number,
                            image_index=global_image_index,
                            image_bytes=png_bytes,
                            width_px=w,
                            height_px=h,
                            bbox=bbox,
                            xref=xref,
                        ))
                        seen_xrefs.add(xref)
                        global_image_index += 1

                    except Exception as exc:
                        logger.debug(
                            "Failed to extract image xref=%d page=%d: %s",
                            xref, page_number, exc,
                        )
                        continue

                # --- Fallback: render full page if no raster images found
                #     but page likely has vector diagrams (non-text content) ---
                if not page_images and len(page_images) < max_per_page:
                    page_images += _try_render_page(
                        page,
                        page_number=page_number,
                        start_index=global_image_index,
                        min_px=min_px,
                    )
                    global_image_index += len(page_images)

            except Exception as exc:
                logger.warning(
                    "Image extraction failed for page %d: %s", page_number, exc
                )
                continue

            results.extend(page_images)

    finally:
        doc.close()

    logger.info(
        "Extracted %d image(s) from %s", len(results), path.name
    )
    return results


def _to_png(raw_bytes: bytes, ext: str) -> bytes | None:
    """
    Convert raw image bytes to PNG. If already PNG/JPEG-like, attempt
    re-encoding via Pillow. Falls back to raw bytes if Pillow not available.
    """
    try:
        from PIL import Image as PILImage

        buf = io.BytesIO(raw_bytes)
        img = PILImage.open(buf).convert("RGB")
        out = io.BytesIO()
        img.save(out, format="PNG", optimize=False)
        return out.getvalue()
    except Exception:
        # If Pillow is not available or conversion fails, return raw bytes
        # (Ollama's /api/chat accepts JPEG too)
        return raw_bytes if raw_bytes else None


def _get_image_bbox(page: Any, xref: int) -> dict[str, float] | None:
    """Extract the bounding box of a specific image on a PDF page."""
    try:
        for block in page.get_text("rawdict")["blocks"]:
            if block.get("type") == 1:  # image block
                if block.get("number") == xref:
                    r = block["bbox"]
                    return {"x0": r[0], "y0": r[1], "x1": r[2], "y1": r[3]}
    except Exception:
        pass
    return None


def _try_render_page(
    page: Any,
    page_number: int,
    start_index: int,
    min_px: int,
) -> list[ExtractedImage]:
    """
    Render a full page as a PNG raster at 150 DPI.
    Only used as fallback when no embedded images are found but the page has
    non-trivial content (measured by page area vs text coverage).
    """
    try:
        import fitz

        # Only render pages with meaningful non-text content
        text = page.get_text("text")
        total_text_chars = len(text.strip())
        # Heuristic: if the page has very little text but non-zero area, it's likely a diagram
        page_area = page.rect.width * page.rect.height
        text_density = total_text_chars / max(page_area, 1)
        if text_density > 0.5 or total_text_chars > 2000:
            # Text-heavy page — skip full-page render
            return []

        mat = fitz.Matrix(150 / 72, 150 / 72)  # 150 DPI
        pix = page.get_pixmap(matrix=mat, alpha=False)
        if pix.width < min_px or pix.height < min_px:
            return []

        png_bytes = pix.tobytes("png")
        logger.debug(
            "Rendered full page %d as fallback image (%dx%d px)",
            page_number, pix.width, pix.height,
        )
        return [ExtractedImage(
            page_number=page_number,
            image_index=start_index,
            image_bytes=png_bytes,
            width_px=pix.width,
            height_px=pix.height,
            bbox=None,
            xref=None,
        )]
    except Exception as exc:
        logger.debug("Full-page render failed for page %d: %s", page_number, exc)
        return []
