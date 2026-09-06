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
class TableItem:
    title: str
    markdown: str
    page_number: int
    section: str = ""
    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)


@dataclass
class PageContent:
    page_number: int
    text: str
    headings: list[str] = field(default_factory=list)
    tables: list[TableItem] = field(default_factory=list)


KNOWN_TABLE_HEADERS = {
    # Weapons & Combat
    "WAFFE",
    "WEAPON",
    "WEAPONS",
    "ART",
    "TYPE",
    "SCHADEN",
    "DAMAGE",
    "DV",
    "MODUS",
    "MODE",
    "ANGRIFFSWERT",
    "ANGRIFFSWERTE",
    "ATTACK RATING",
    "MUNITION",
    "AMMO",
    "VERFÜGBARKEIT",
    "VERFÜGB.",
    "VERF.",
    "AVAIL",
    "AVAILABILITY",
    "PREIS",
    "PRICE",
    "KOSTEN",
    "COST",
    "REICHWEITE",
    "RANGE",
    "GRANATE",
    "GRENADE",
    "SPRENGWIRKUNG",
    "BLAST",
    "RADIUS",
    "ZUBEHÖR",
    "ACCESSORY",
    "HALTERUNG",
    "MOUNT",
    "ENERGIEEINHEITEN",
    # Armor & Defense
    "RÜSTUNG",
    "ARMOR",
    "PANZERUNG",
    "PANZ.",
    "V.-WERT",
    "VERTEIDIGUNGSWERT",
    "DEFENSE RATING",
    "SOZIAL",
    "SOCIAL",
    # Magic, Matrix, Equipment
    "KAPAZITÄT",
    "KAP.",
    "CAPACITY",
    "STUFE",
    "LEVEL",
    "RATING",
    "WÜRFELPOOL",
    "DICE POOL",
    "HANDLUNG",
    "ACTION",
    "HANDLUNGSART",
    "DAUER",
    "DURATION",
    "ENTZUG",
    "DRAIN",
    "SLOTS",
    "PROGRAMMSLOTS",
    "GS",
    "DEVICE RATING",
    "ASDF",
    "AUFLÖSUNG",
    "AUSDEHNUNG",
    "OPTION",
    "GEGENSTAND",
    "ITEM",
    "NAME",
    "TYP",
    "WOFÜR?",
    # Attributes, Stats & Mechanics
    "KONSTITUTION",
    "GESCHICKLICHKEIT",
    "REAKTION",
    "STÄRKE",
    "WILLENSKRAFT",
    "LOGIK",
    "INTUITION",
    "CHARISMA",
    "EDGE",
    "ESSENZ",
    "MAGIE",
    "STR",
    "DEX",
    "CON",
    "INT",
    "WIS",
    "CHA",
    "HP",
    "AC",
    "CR",
    "METAVARIANTE",
    "FAHNDUNGSSTUFE",
    "AUSWIRKUNG",
    "KARMAKOSTEN",
    "TRAININGSZEIT",
    "REPUTATION",
    "REPUTATIONS-ÄNDERUNG",
    "SENSOR",
    "PILOT",
    "RUMPF",
    "BODY",
    "SPEED",
    "BESCHL.",
    "GESCHW.",
    "HÖCHSTG.",
    "G.INTV.",
    "HANDL.",
    "HANDL. (S/G)",
}


def _is_header_candidate(line: str) -> bool:
    line_clean = line.strip().upper().replace(":", "").replace("*", "")
    if line_clean in KNOWN_TABLE_HEADERS:
        return True
    words = [w.strip() for w in line_clean.split()]
    if any(w in KNOWN_TABLE_HEADERS for w in words):
        return True
    return bool(line.isupper() and 1 < len(line) < 25 and not any(c.isdigit() for c in line))


def _parse_block_table(lines: list[str]) -> tuple[list[str], list[list[str]]] | None:
    if len(lines) < 4:
        return None

    headers: list[str] = []
    for line in lines:
        if _is_header_candidate(line):
            headers.append(line)
        else:
            break

    if len(headers) < 2:
        return None

    if not any(
        h.strip().upper().replace(":", "").replace("*", "") in KNOWN_TABLE_HEADERS
        or any(w in KNOWN_TABLE_HEADERS for w in h.strip().upper().split())
        for h in headers
    ):
        return None

    num_cols = len(headers)
    data = lines[num_cols:]
    if not data:
        return None

    num_rows = len(data) // num_cols
    if num_rows < 1:
        return None

    rows: list[list[str]] = []
    for r in range(num_rows):
        rows.append(data[r * num_cols : (r + 1) * num_cols])
    remainder = data[num_rows * num_cols :]
    if remainder:
        padded = remainder + ["–"] * (num_cols - len(remainder))
        rows.append(padded)

    return headers, rows


def _format_markdown_table(title: str, headers: list[str], rows: list[list[str]]) -> str:
    out: list[str] = []
    if title:
        out.append(f"### Tabelle: {title}")
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        clean_row = [str(c).replace("|", "/").replace("\n", " ").strip() for c in row]
        out.append("| " + " | ".join(clean_row) + " |")
    return "\n".join(out)


def _extract_tables(page: pymupdf.Page, page_number: int, headings: list[str]) -> list[TableItem]:
    tables: list[TableItem] = []
    tab_rects: list[pymupdf.Rect] = []

    # 1. Built-in find_tables for bordered/grid tables
    try:
        page_tabs = page.find_tables()
        if page_tabs and page_tabs.tables:
            for tab in page_tabs.tables:
                extracted = tab.extract()
                if not extracted or len(extracted) < 2:
                    continue
                raw_headers = [str(c).strip().replace("\n", " ") if c else "" for c in extracted[0]]
                raw_rows = [
                    [str(c).strip().replace("\n", " ") if c else "" for c in row]
                    for row in extracted[1:]
                ]
                if len(raw_headers) >= 2 and any(raw_headers) and raw_rows:
                    title = headings[-1] if headings else ""
                    md = _format_markdown_table(title, raw_headers, raw_rows)
                    tables.append(
                        TableItem(
                            title=title,
                            markdown=md,
                            page_number=page_number,
                            section=title,
                            headers=raw_headers,
                            rows=raw_rows,
                        )
                    )
                    tab_rects.append(pymupdf.Rect(tab.bbox))
    except Exception:
        pass

    # 2. Block-based table detection for borderless RPG tables
    blocks = page.get_text("blocks")
    for i, b in enumerate(blocks):
        b_rect = pymupdf.Rect(b[:4])
        if any(b_rect.intersects(tr) for tr in tab_rects):
            continue

        lines = [line_item.strip() for line_item in b[4].split("\n") if line_item.strip()]
        res = _parse_block_table(lines)
        if res:
            headers, rows = res
            title = ""
            if i > 0 and len(blocks[i - 1][4].strip().split("\n")) <= 2:
                title = blocks[i - 1][4].strip()
            if not title and headings:
                title = headings[-1]

            md = _format_markdown_table(title, headers, rows)
            tables.append(
                TableItem(
                    title=title,
                    markdown=md,
                    page_number=page_number,
                    section=title,
                    headers=headers,
                    rows=rows,
                )
            )

    return tables


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
    text = page.get_text("dict", sort=False)
    if not isinstance(text, dict):
        return []
    blocks = text["blocks"]

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

    for page_num in range(doc.page_count):
        page = doc[page_num]
        # Use plain text extraction which handles columns correctly
        text = page.get_text("text")
        if not isinstance(text, str):
            continue
        full_text = _dedup_layered_text(text)
        if not full_text.strip():
            continue

        # Use dict extraction only for heading detection
        headings = _extract_headings(page)
        tables = _extract_tables(page, page_num + 1, headings)

        pages.append(
            PageContent(
                page_number=page_num + 1,
                text=full_text,
                headings=headings,
                tables=tables,
            )
        )

    doc.close()
    return pages
