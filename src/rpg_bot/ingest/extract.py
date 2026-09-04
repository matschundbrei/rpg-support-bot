from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

# PyMuPDF's macOS wheels are built with SWIG 4.3.1, whose generated
# types lack __module__ and trigger these harmless import-time
# warnings (pymupdf/PyMuPDF#3931).
warnings.filterwarnings(
    "ignore",
    message=r"builtin type (SwigPyPacked|SwigPyObject|swigvarlink) has no __module__ attribute",
)

import pymupdf  # noqa: E402  # must follow warnings.filterwarnings above


@dataclass
class PageContent:
    page_number: int
    text: str
    headings: list[str] = field(default_factory=list)


def _is_heading(span: dict, page_avg_size: float) -> bool:
    """Heuristic: a span is a heading if it's significantly larger than average body text."""
    size = span.get("size", 0)
    is_large = size > page_avg_size * 1.35
    text = span.get("text", "").strip()
    if not text or len(text) > 80 or len(text) < 3:
        return False
    # Skip page numbers, pure digits, ellipsis lines (TOC)
    if text.replace(".", "").replace(" ", "").replace("/", "").isdigit():
        return False
    if "..." in text or "…" in text:
        return False
    return is_large


def _extract_headings(page: pymupdf.Page) -> list[str]:
    """Extract headings from a page using font size heuristics."""
    blocks = page.get_text("dict", sort=False)["blocks"]

    all_sizes: list[float] = []
    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if span.get("text", "").strip():
                    all_sizes.append(span["size"])

    avg_size = sum(all_sizes) / len(all_sizes) if all_sizes else 12.0

    seen: set[str] = set()
    headings: list[str] = []
    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if _is_heading(span, avg_size):
                    text = span["text"].strip()
                    if text not in seen:
                        seen.add(text)
                        headings.append(text)

    return headings


def _dedup_layered_text(text: str) -> str:
    """Remove duplicate text from PDFs with layered/overlapping content.

    Some PDFs (e.g. those with background images) store text in multiple
    layers, causing get_text() to return the same content twice. If the
    second half of the text repeats the first half, keep only one copy.
    """
    lines = text.split("\n")
    if len(lines) < 10:
        return text

    first_line = lines[0].strip()
    if not first_line:
        return text

    # Find if the first line repeats near the middle of the text
    for i in range(len(lines) // 3, 2 * len(lines) // 3 + 1):
        if lines[i].strip() == first_line:
            # Verify it's truly a duplicate by checking a few more lines
            first_half = [line.strip() for line in lines[:5] if line.strip()]
            second_half = [line.strip() for line in lines[i : i + 5] if line.strip()]
            if first_half == second_half:
                return "\n".join(lines[:i])

    return text


def extract_pdf(pdf_path: Path) -> list[PageContent]:
    """Extract text from a PDF, handling multi-column layouts."""
    doc = pymupdf.open(str(pdf_path))
    pages: list[PageContent] = []

    for page_num, page in enumerate(doc, start=1):
        # Use plain text extraction which handles columns correctly
        full_text = _dedup_layered_text(page.get_text("text"))
        if not full_text.strip():
            continue

        # Use dict extraction only for heading detection
        headings = _extract_headings(page)

        pages.append(
            PageContent(
                page_number=page_num,
                text=full_text,
                headings=headings,
            )
        )

    doc.close()
    return pages
