from __future__ import annotations

import re
from dataclasses import dataclass, field

from rpg_bot.config import get_settings
from rpg_bot.ingest.extract import PageContent


@dataclass
class Chunk:
    text: str
    metadata: dict[str, str | int] = field(default_factory=dict)


def _detect_language(text: str) -> str:
    try:
        from langdetect import detect

        return detect(text[:500])
    except Exception:
        return "en"


def _build_breadcrumb(heading: str, source_name: str) -> str:
    if heading:
        return f"{source_name} > {heading}"
    return source_name


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Recursive character split with overlap, preferring paragraph/sentence boundaries."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    separators = ["\n\n", "\n", ". ", " "]

    def _do_split(t: str, sep_idx: int = 0) -> list[str]:
        if len(t) <= chunk_size or sep_idx >= len(separators):
            return [t] if t.strip() else []

        sep = separators[sep_idx]
        parts = t.split(sep)
        result: list[str] = []
        current = ""

        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    result.append(current)
                if len(part) > chunk_size:
                    result.extend(_do_split(part, sep_idx + 1))
                else:
                    current = part
        if current.strip():
            result.append(current)
        return result

    raw_chunks = _do_split(text)

    # Apply overlap
    chunks: list[str] = []
    for i, chunk in enumerate(raw_chunks):
        if i > 0 and overlap > 0:
            prev = raw_chunks[i - 1]
            overlap_text = prev[-overlap:] if len(prev) > overlap else prev
            space_idx = overlap_text.find(" ")
            if space_idx > 0:
                overlap_text = overlap_text[space_idx + 1 :]
            chunk = overlap_text + " " + chunk
        chunks.append(chunk.strip())

    return [c for c in chunks if c]


_MIN_SECTION_SIZE = 150


def _split_into_sections(text: str, headings: list[str]) -> list[tuple[str, str]]:
    """Split page text at heading boundaries.

    Returns list of (heading, section_text) tuples. Merges tiny sections
    with the next section to avoid chunks too small for good embeddings.
    """
    # Find heading positions (must appear at start of a line)
    positions: list[tuple[int, str]] = []
    for heading in headings:
        pattern = re.compile(r"^" + re.escape(heading) + r"\s*$", re.MULTILINE)
        match = pattern.search(text)
        if match and match.start() not in {p for p, _ in positions}:
            positions.append((match.start(), heading))

    if not positions:
        return [("", text)]

    positions.sort(key=lambda x: x[0])

    # Build raw sections
    raw: list[tuple[str, str]] = []
    pre_text = text[: positions[0][0]].strip()
    if pre_text:
        raw.append(("", pre_text))

    for i, (pos, heading) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        section_text = text[pos:end].strip()
        if section_text:
            raw.append((heading, section_text))

    # Merge tiny sections into their predecessor to avoid polluting
    # the next section's heading/embedding context
    merged: list[tuple[str, str]] = []
    for heading, section_text in raw:
        if merged and len(section_text) < _MIN_SECTION_SIZE:
            prev_h, prev_t = merged[-1]
            merged[-1] = (prev_h, prev_t + "\n\n" + section_text)
        else:
            merged.append((heading, section_text))

    return merged


def chunk_pages(
    pages: list[PageContent],
    source_name: str,
    game_system: str = "",
    source_path: str = "",
) -> list[Chunk]:
    """Chunk extracted pages into retrieval-ready chunks with metadata."""
    settings = get_settings()
    chunk_size = settings.chunking.chunk_size
    overlap = settings.chunking.chunk_overlap

    # Detect language from first pages
    sample_text = " ".join(p.text[:200] for p in pages[:5])
    language = _detect_language(sample_text)

    all_chunks: list[Chunk] = []

    for page in pages:
        sections = _split_into_sections(page.text, page.headings)

        # Fallback heading when no headings were found in the text
        fallback_heading = page.headings[-1] if page.headings else ""

        for section_heading, section_text in sections:
            heading = section_heading or fallback_heading
            breadcrumb = _build_breadcrumb(heading, source_name)

            # Split large sections further; keep small ones as-is
            if len(section_text) <= chunk_size:
                text_chunks = [section_text]
            else:
                text_chunks = _split_text(section_text, chunk_size, overlap)

            for text in text_chunks:
                chunk_text = f"[{breadcrumb}]\n{text}"

                all_chunks.append(
                    Chunk(
                        text=chunk_text,
                        metadata={
                            "source": source_name,
                            "source_path": source_path,
                            "page": page.page_number,
                            "game_system": game_system,
                            "language": language,
                            "section": section_heading,
                            "breadcrumb": breadcrumb,
                        },
                    )
                )

    return all_chunks
