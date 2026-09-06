from rpg_bot.ingest.chunker import _chunk_table, _split_text, chunk_pages
from rpg_bot.ingest.extract import PageContent, TableItem, _parse_block_table


def test_split_text_short():
    """Short text should not be split."""
    result = _split_text("Hello world", chunk_size=1000, overlap=200)
    assert result == ["Hello world"]


def test_split_text_respects_size():
    """All chunks should be roughly within chunk_size."""
    text = "Word " * 500  # ~2500 chars
    chunks = _split_text(text, chunk_size=500, overlap=0)
    assert len(chunks) > 1
    for chunk in chunks:
        # Allow some slack for overlap and boundary handling
        assert len(chunk) < 700


def test_split_text_overlap():
    """Chunks should overlap when overlap > 0."""
    text = "Paragraph one. " * 30 + "\n\n" + "Paragraph two. " * 30
    chunks = _split_text(text, chunk_size=200, overlap=50)
    assert len(chunks) > 1
    # With overlap, later chunks should contain some text from the previous chunk
    if len(chunks) >= 2:
        # With overlap, the tail of the previous chunk reappears at the start
        # of the next one (trimmed to a word boundary)
        assert chunks[0][-30:].rstrip() in chunks[1]


def test_chunk_pages_metadata():
    """Chunks should have correct metadata."""
    pages = [
        PageContent(
            page_number=1,
            text="The fighter attacks with a longsword dealing 1d8 slashing damage.",
            headings=["Combat", "Melee Attacks"],
        ),
        PageContent(
            page_number=2,
            text="Spellcasting requires concentration. If hit, make a Constitution save.",
            headings=["Spellcasting"],
        ),
    ]
    chunks = chunk_pages(pages, source_name="PHB", game_system="dnd5e")
    assert len(chunks) >= 2

    for chunk in chunks:
        assert chunk.metadata["source"] == "PHB"
        assert chunk.metadata["game_system"] == "dnd5e"
        assert "language" in chunk.metadata
        assert chunk.metadata["page"] in (1, 2)

    # First chunk should have breadcrumb
    assert "[PHB" in chunks[0].text


def test_chunk_pages_breadcrumb():
    """Breadcrumbs should include source and section headings."""
    pages = [
        PageContent(
            page_number=42,
            text="Opportunity attacks occur when...",
            headings=["Chapter 9", "Combat", "Opportunity Attacks"],
        ),
    ]
    chunks = chunk_pages(pages, source_name="PHB")
    assert chunks
    assert "PHB > Opportunity Attacks" in chunks[0].text


def test_parse_block_table():
    """Block table parser should extract headers and data rows."""
    lines = [
        "WAFFE",
        "SCHADEN",
        "PREIS",
        "Ares Predator VI",
        "3K",
        "750 ¥",
        "Ruger Super Warhawk",
        "4K",
        "400 ¥",
    ]
    res = _parse_block_table(lines)
    assert res is not None
    headers, rows = res
    assert headers == ["WAFFE", "SCHADEN", "PREIS"]
    assert len(rows) == 2
    assert rows[0] == ["Ares Predator VI", "3K", "750 ¥"]
    assert rows[1] == ["Ruger Super Warhawk", "4K", "400 ¥"]


def test_chunk_pages_with_tables():
    """Table chunks should be extracted with table metadata and markdown structure."""
    table = TableItem(
        title="Schwere Pistolen",
        markdown=(
            "### Tabelle: Schwere Pistolen\n"
            "| WAFFE | SCHADEN |\n"
            "| --- | --- |\n"
            "| Ares Predator | 3K |"
        ),
        page_number=255,
        section="Schwere Pistolen",
        headers=["WAFFE", "SCHADEN"],
        rows=[["Ares Predator", "3K"]],
    )
    pages = [
        PageContent(
            page_number=255,
            text="Pistolen sind handliche Feuerwaffen.",
            headings=["Feuerwaffen", "Schwere Pistolen"],
            tables=[table],
        ),
    ]
    chunks = chunk_pages(pages, source_name="SR6", game_system="shadowrun6")
    # Expect 1 text chunk + 1 table chunk
    assert len(chunks) == 2

    table_chunk = next(c for c in chunks if c.metadata.get("is_table") == "true")
    assert table_chunk.metadata["content_type"] == "table"
    assert table_chunk.metadata["page"] == 255
    assert "Tabelle: Schwere Pistolen" in table_chunk.text
    assert "| Ares Predator | 3K |" in table_chunk.text


def test_chunk_table_preserves_headers():
    """Large tables split into multiple chunks must preserve column headers in every chunk."""
    headers = ["WAFFE", "SCHADEN", "PREIS"]
    rows = [[f"Waffe {i}", f"{i}K", f"{i * 100} ¥"] for i in range(20)]
    table = TableItem(
        title="Waffenliste",
        markdown="",
        page_number=10,
        headers=headers,
        rows=rows,
    )
    chunks = _chunk_table(table, chunk_size=200)
    assert len(chunks) > 1
    for chunk in chunks:
        assert "| WAFFE | SCHADEN | PREIS |" in chunk
        assert "| --- | --- | --- |" in chunk
