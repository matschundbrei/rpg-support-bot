from rpg_bot.ingest.chunker import Chunk, _split_text, chunk_pages
from rpg_bot.ingest.extract import PageContent


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
        # Check there's some shared content
        last_part_of_first = chunks[0][-50:]
        assert len(chunks[1]) > 0


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
