"""
app/services/image_store.py

Filesystem helpers for extracted image files.

Images are stored at:
  {IMAGES_DIR}/{tenant_id}/{doc_id}/page_{n}_img_{i}.png

This path is deterministic given (tenant_id, doc_id, page_number, image_index),
so it can be reconstructed from DB metadata without querying the DB.

The /images/files/ static mount in main.py serves these files over HTTP.
The URL path is:
  /images/files/{tenant_id}/{doc_id}/page_{n}_img_{i}.png
"""
from __future__ import annotations

import logging
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


def _image_filename(page_number: int | None, image_index: int) -> str:
    page_str = f"page_{page_number}" if page_number is not None else "page_unknown"
    return f"{page_str}_img_{image_index}.png"


def get_image_path(
    tenant_id: str,
    doc_id: str,
    page_number: int | None,
    image_index: int,
) -> Path:
    """Compute the deterministic storage path for an extracted image."""
    return (
        Path(settings.images_dir)
        / str(tenant_id)
        / str(doc_id)
        / _image_filename(page_number, image_index)
    )


def save_image(
    image_bytes: bytes,
    tenant_id: str,
    doc_id: str,
    page_number: int | None,
    image_index: int,
) -> Path:
    """
    Write image bytes to the deterministic path.
    Creates parent directories as needed.
    Returns the absolute Path where the file was saved.
    """
    path = get_image_path(tenant_id, doc_id, page_number, image_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(image_bytes)
    logger.debug("Saved image: %s (%d bytes)", path, len(image_bytes))
    return path


def load_image(storage_path: str) -> bytes:
    """
    Read image bytes from disk.
    Raises FileNotFoundError if the path does not exist.
    """
    path = Path(storage_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found on disk: {storage_path}")
    return path.read_bytes()


def storage_path_to_url(storage_path: str, base_url: str = "") -> str:
    """
    Convert an absolute filesystem path to a resolvable HTTP URL.

    The static mount serves {IMAGES_DIR} at /images/files/.
    Example:
      /app/data/images/tenant-uuid/doc-uuid/page_1_img_0.png
      → /images/files/tenant-uuid/doc-uuid/page_1_img_0.png
    """
    images_dir = Path(settings.images_dir)
    try:
        relative = Path(storage_path).relative_to(images_dir)
        url_path = f"/images/files/{relative}"
        return f"{base_url.rstrip('/')}{url_path}" if base_url else url_path
    except ValueError:
        # storage_path is not under images_dir — return as-is
        return storage_path
