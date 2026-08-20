"""
app/worker/chunker.py

Structure-aware chunker for the ingestion pipeline.

Strategy (in order of precedence):
  1. Markdown headings (ATX: #, ##, ###) — split text into sections.
  2. PDF heading heuristics — use headings extracted by pdfplumber.
  3. Token-based sliding window fallback — when no headings are detected.

Within each section, we recursively split to stay within target_tokens,
maintaining chunk_overlap_tokens of overlap between consecutive chunks.

Token counting uses tiktoken (cl100k_base) — fast, accurate, no network calls.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from app.config import settings
from app.worker.parsers import ParsedDocument, ParsedPage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tiktoken setup (lazy import to avoid slowing startup if unused)
# ---------------------------------------------------------------------------

def _get_encoder():
    try:
        import tiktoken  # type: ignore[import]
        return tiktoken.get_encoding("cl100k_base")
    except ImportError as exc:
        raise RuntimeError("tiktoken is required for token counting.") from exc


# Markdown ATX heading pattern
_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    content: str
    content_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core chunker class
# ---------------------------------------------------------------------------

class StructureAwareChunker:
    """
    Produces chunks from a ParsedDocument.

    Args:
        target_tokens: Desired maximum tokens per chunk (default from config).
        overlap_tokens: Token overlap between consecutive chunks.
    """

    def __init__(
        self,
        target_tokens: int | None = None,
        overlap_tokens: int | None = None,
    ) -> None:
        self._enc = _get_encoder()
        self.target = target_tokens or settings.chunk_target_tokens
        self.overlap = overlap_tokens or settings.chunk_overlap_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_document(self, doc: ParsedDocument) -> list[Chunk]:
        """
        Chunk an entire ParsedDocument into a flat list of Chunk objects.
        """
        all_chunks: list[Chunk] = []
        global_chunk_index = 0

        for page in doc.pages:
            page_chunks = self._chunk_page(page, doc.source_type)
            for chunk in page_chunks:
                chunk.metadata["chunk_index"] = global_chunk_index
                global_chunk_index += 1
                all_chunks.append(chunk)

        logger.info("Chunked document into %d chunk(s).", len(all_chunks))
        return all_chunks

    # ------------------------------------------------------------------
    # Per-page chunking
    # ------------------------------------------------------------------

    def _chunk_page(self, page: ParsedPage, source_type: str) -> list[Chunk]:
        if source_type == "markdown":
            return self._chunk_markdown(page)
        if source_type == "pdf" and page.headings:
            return self._chunk_pdf_with_headings(page)
        # Fallback: sliding-window on raw text
        return self._sliding_window(
            text=page.text,
            base_metadata={"page_number": page.page_number},
            section_path=None,
        )

    # ------------------------------------------------------------------
    # Markdown: split on heading boundaries, then slide within each section
    # ------------------------------------------------------------------

    def _chunk_markdown(self, page: ParsedPage) -> list[Chunk]:
        text = page.text
        sections = self._split_markdown_sections(text)
        chunks: list[Chunk] = []

        for (section_path, section_text) in sections:
            chunks.extend(
                self._sliding_window(
                    text=section_text,
                    base_metadata={"page_number": page.page_number},
                    section_path=section_path,
                )
            )
        return chunks

    def _split_markdown_sections(self, text: str) -> list[tuple[str, str]]:
        """
        Split Markdown text into (section_path, text) pairs by ATX headings.
        Falls back to a single section if no headings are found.
        """
        matches = list(_MD_HEADING_RE.finditer(text))
        if not matches:
            return [("", text)]

        sections: list[tuple[str, str]] = []
        heading_stack: list[tuple[int, str]] = []  # (level, text)

        for i, match in enumerate(matches):
            level = len(match.group(1))
            heading_text = match.group(2).strip()

            # Content between this heading and the next
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()

            # Build breadcrumb: pop stack to current level, push new heading
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading_text))
            section_path = " > ".join(h for _, h in heading_stack)

            # Include some text before the first heading
            if i == 0 and matches[0].start() > 0:
                preamble = text[: matches[0].start()].strip()
                if preamble:
                    sections.append(("", preamble))

            if content:
                sections.append((section_path, content))

        return sections if sections else [("", text)]

    # ------------------------------------------------------------------
    # PDF with headings: treat heading list as section boundaries
    # ------------------------------------------------------------------

    def _chunk_pdf_with_headings(self, page: ParsedPage) -> list[Chunk]:
        """
        For PDFs, use headings extracted by pdfplumber to split the page text
        into sections, then sliding-window within each section.
        Falls back to plain sliding window if sectioning fails.
        """
        text = page.text
        if not page.headings:
            return self._sliding_window(
                text=text,
                base_metadata={"page_number": page.page_number},
                section_path=None,
            )

        chunks: list[Chunk] = []
        remaining = text

        for j, heading in enumerate(page.headings):
            idx = remaining.find(heading)
            if idx == -1:
                continue
            before = remaining[:idx].strip()
            if before:
                path = page.headings[j - 1] if j > 0 else ""
                chunks.extend(
                    self._sliding_window(
                        text=before,
                        base_metadata={"page_number": page.page_number},
                        section_path=path,
                    )
                )
            remaining = remaining[idx + len(heading):]

        # Tail
        if remaining.strip():
            chunks.extend(
                self._sliding_window(
                    text=remaining,
                    base_metadata={"page_number": page.page_number},
                    section_path=page.headings[-1] if page.headings else None,
                )
            )

        return chunks if chunks else self._sliding_window(
            text=text,
            base_metadata={"page_number": page.page_number},
            section_path=None,
        )

    # ------------------------------------------------------------------
    # Token-based sliding window
    # ------------------------------------------------------------------

    def _sliding_window(
        self,
        text: str,
        base_metadata: dict[str, Any],
        section_path: str | None,
    ) -> list[Chunk]:
        """
        Split text into overlapping token windows.
        Produces chunks in [target - overlap, target] token range.
        """
        text = text.strip()
        if not text:
            return []

        tokens = self._enc.encode(text)
        if len(tokens) <= self.target:
            # Text fits in one chunk
            return [self._make_chunk(text, base_metadata, section_path)]

        chunks: list[Chunk] = []
        start = 0
        step = self.target - self.overlap

        while start < len(tokens):
            end = min(start + self.target, len(tokens))
            window_tokens = tokens[start:end]
            window_text = self._enc.decode(window_tokens)
            chunks.append(self._make_chunk(window_text, base_metadata, section_path))
            if end == len(tokens):
                break
            start += step

        return chunks

    # ------------------------------------------------------------------
    # Chunk factory
    # ------------------------------------------------------------------

    def _make_chunk(
        self,
        text: str,
        base_metadata: dict[str, Any],
        section_path: str | None,
    ) -> Chunk:
        content = text.strip()
        metadata: dict[str, Any] = {**base_metadata}
        if section_path:
            metadata["section_path"] = section_path

        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        return Chunk(
            content=content,
            content_hash=content_hash,
            metadata=metadata,
        )
