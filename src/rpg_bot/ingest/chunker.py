from __future__ import annotations

import re
from dataclasses import dataclass, field

from rpg_bot.config import get_settings
from rpg_bot.ingest.extract import PageContent


@dataclass
class Chunk:
    text: str
    metadata: dict[str, str | int] = field(default_factory=dict)


# Patterns that indicate section boundaries in RPG books
_SECTION_PATTERNS = [
    re.compile(r"^#{1,3}\s+", re.MULTILINE),  # Markdown-style
    re.compile(r"^(?:Chapter|Part|Section|Kapitel|Teil|Abschnitt)\s+\d", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^[A-Z][A-Z\s]{4,}$", re.MULTILINE),  # ALL CAPS headings
]

# Patterns for atomic RPG content that should not be split
_STAT_BLOCK_START = re.compile(
    r"(?:^|\n)(?:"
    r"(?:Armor Class|Hit Points|Speed|STR|DEX|CON|INT|WIS|CHA)"
    r"|(?:Rüstungsklasse|Trefferpunkte|Bewegungsrate)"
    r")",
    re.IGNORECASE,
)


def _detect_language(text: str) -> str:
    try:
        from langdetect import detect
        return detect(text[:500])
    except Exception:
        return "en"


def _build_breadcrumb(headings: list[str], source_name: str) -> str:
    parts = [source_name]
    # Take last 2 unique headings as breadcrumb path
    seen = set()
    for h in headings[-2:]:
        h_clean = h.strip()
        if h_clean and h_clean.lower() not in seen:
            seen.add(h_clean.lower())
            parts.append(h_clean)
    return " > ".join(parts)


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Recursive character split with overlap, preferring paragraph/sentence boundaries."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    chunks: list[str] = []
    # Try splitting on paragraph breaks first, then sentences, then hard split
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
    for i, chunk in enumerate(raw_chunks):
        if i > 0 and overlap > 0:
            prev = raw_chunks[i - 1]
            overlap_text = prev[-overlap:] if len(prev) > overlap else prev
            # Find a clean break point in the overlap
            space_idx = overlap_text.find(" ")
            if space_idx > 0:
                overlap_text = overlap_text[space_idx + 1:]
            chunk = overlap_text + " " + chunk
        chunks.append(chunk.strip())

    return [c for c in chunks if c]


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
    active_headings: list[str] = []

    for page in pages:
        # Update heading context
        if page.headings:
            active_headings = page.headings

        breadcrumb = _build_breadcrumb(active_headings, source_name)

        # Split page text into chunks
        text_chunks = _split_text(page.text, chunk_size, overlap)

        for text in text_chunks:
            # Prepend breadcrumb for context
            chunk_text = f"[{breadcrumb}]\n{text}"

            all_chunks.append(Chunk(
                text=chunk_text,
                metadata={
                    "source": source_name,
                    "source_path": source_path,
                    "page": page.page_number,
                    "game_system": game_system,
                    "language": language,
                    "section": active_headings[-1] if active_headings else "",
                    "breadcrumb": breadcrumb,
                },
            ))

    return all_chunks
