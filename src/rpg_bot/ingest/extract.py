from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pymupdf


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


def extract_pdf(pdf_path: Path) -> list[PageContent]:
    """Extract text from a PDF, handling multi-column layouts via block sorting."""
    doc = pymupdf.open(str(pdf_path))
    pages: list[PageContent] = []

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict", sort=True)["blocks"]

        page_text_parts: list[str] = []
        headings: list[str] = []
        all_sizes: list[float] = []

        # First pass: collect font sizes for average calculation
        for block in blocks:
            if block.get("type") != 0:  # text blocks only
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span.get("text", "").strip():
                        all_sizes.append(span["size"])

        avg_size = sum(all_sizes) / len(all_sizes) if all_sizes else 12.0

        # Second pass: extract text and detect headings
        for block in blocks:
            if block.get("type") != 0:
                continue
            block_text_parts: list[str] = []
            for line in block.get("lines", []):
                line_text = ""
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    line_text += text
                    if _is_heading(span, avg_size) and text.strip():
                        headings.append(text.strip())
                block_text_parts.append(line_text)

            block_text = "\n".join(block_text_parts).strip()
            if block_text:
                page_text_parts.append(block_text)

        full_text = "\n\n".join(page_text_parts)
        if full_text.strip():
            pages.append(PageContent(
                page_number=page_num,
                text=full_text,
                headings=headings,
            ))

    doc.close()
    return pages
