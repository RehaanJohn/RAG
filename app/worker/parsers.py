"""
app/worker/parsers.py

File parsers for each supported source type.
Returns structured output so the chunker knows page numbers and structure.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ParsedPage:
    """
    A single page or logical section of a parsed document.
    For plain text / Markdown, the whole file is a single page (page_number=1).
    """
    page_number: int
    text: str
    # Headings extracted at the page level (for PDF font-size heuristics)
    headings: list[str] = field(default_factory=list)


@dataclass
class ParsedDocument:
    pages: list[ParsedPage]
    source_type: str


# ---------------------------------------------------------------------------
# PDF parser — pdfplumber
# ---------------------------------------------------------------------------

def parse_pdf(path: Path) -> ParsedDocument:
    """
    Extract text from a PDF page-by-page using pdfplumber.
    Attempts basic heading detection via font-size differences.
    """
    try:
        import pdfplumber  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError("pdfplumber is required for PDF parsing.") from exc

    pages: list[ParsedPage] = []

    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                logger.debug("PDF page %d is empty — skipping.", i)
                continue

            # Heading heuristic: words/lines whose font size is >= 1.2× median
            headings: list[str] = []
            try:
                words = page.extract_words(extra_attrs=["size"])
                if words:
                    sizes = [float(w.get("size", 0)) for w in words if w.get("size")]
                    if sizes:
                        median_size = sorted(sizes)[len(sizes) // 2]
                        threshold = median_size * 1.2
                        # Group consecutive large words into heading phrases
                        current_heading: list[str] = []
                        for w in words:
                            if float(w.get("size", 0)) >= threshold:
                                current_heading.append(w["text"])
                            elif current_heading:
                                headings.append(" ".join(current_heading))
                                current_heading = []
                        if current_heading:
                            headings.append(" ".join(current_heading))
            except Exception as exc:
                logger.debug("Heading extraction failed for page %d: %s", i, exc)

            pages.append(ParsedPage(page_number=i, text=text, headings=headings))

    logger.info("PDF parsed: %d pages with text from %s", len(pages), path.name)
    return ParsedDocument(pages=pages, source_type="pdf")


# ---------------------------------------------------------------------------
# Plain text parser
# ---------------------------------------------------------------------------

def parse_text(path: Path) -> ParsedDocument:
    """Read a plain .txt file as a single-page document."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return ParsedDocument(
        pages=[ParsedPage(page_number=1, text=text)],
        source_type="txt",
    )


# ---------------------------------------------------------------------------
# Markdown parser
# ---------------------------------------------------------------------------

def parse_markdown(path: Path) -> ParsedDocument:
    """
    Read a Markdown file. Headings are detected via ATX syntax (# ## ###).
    The full text is kept so the chunker can split on heading boundaries.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")

    # Extract heading texts for the ParsedPage (used as metadata hints)
    headings: list[str] = []
    for line in raw.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            heading_text = stripped.lstrip("#").strip()
            if heading_text:
                headings.append(heading_text)

    return ParsedDocument(
        pages=[ParsedPage(page_number=1, text=raw, headings=headings)],
        source_type="markdown",
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def parse_file(path: Path, source_type: str) -> ParsedDocument:
    """Route to the correct parser based on source_type."""
    if source_type == "pdf":
        return parse_pdf(path)
    if source_type == "txt":
        return parse_text(path)
    if source_type == "markdown":
        return parse_markdown(path)
    raise ValueError(f"Unsupported source_type: {source_type!r}")
