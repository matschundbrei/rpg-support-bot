from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from rpg_bot.config import get_settings
from rpg_bot.embeddings import embed_texts
from rpg_bot.ingest.chunker import chunk_pages
from rpg_bot.ingest.extract import PageContent, extract_pdf
from rpg_bot.retrieval.store import VectorStore, get_store

console = Console()

EMBED_BATCH_SIZE = 16
SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def _guess_game_system(path: Path) -> str:
    """Guess game system from directory structure (e.g. sourcebooks/dnd5e/phb.pdf).

    Uses the configured sourcebooks directory name as the anchor, so custom
    sourcebooks_directory settings keep working.
    """
    settings = get_settings()
    anchor = Path(settings.sourcebooks_directory).name
    parts = path.parts
    try:
        sb_idx = parts.index(anchor)
        if sb_idx + 1 < len(parts) - 1:  # there's a subdirectory
            return parts[sb_idx + 1]
    except ValueError:
        pass
    return ""


def extract_text_file(path: Path) -> list[PageContent]:
    """Extract plain text or markdown files into PageContent."""
    content = path.read_text(encoding="utf-8", errors="replace")
    if not content.strip():
        return []
    headings: list[str] = []
    for line in content.splitlines():
        line_s = line.strip()
        if line_s.startswith("#"):
            heading = line_s.lstrip("#").strip()
            if heading and heading not in headings:
                headings.append(heading)
    return [PageContent(page_number=1, text=content, headings=headings, tables=[])]


def ingest_file(file_path: Path, store: VectorStore) -> int:
    """Ingest a single document file (PDF, Markdown, or text). Returns number of chunks added."""
    file_hash = _file_hash(file_path)

    if store.has_source(file_hash):
        console.print(f"[dim]Skipping (already ingested): {file_path.name}[/dim]")
        return 0

    source_name = file_path.stem
    source_path = str(file_path.resolve())
    game_system = _guess_game_system(file_path)

    console.print(f"[bold]Extracting:[/bold] {file_path.name}")
    if file_path.suffix.lower() == ".pdf":
        pages = extract_pdf(file_path)
    elif file_path.suffix.lower() in [".md", ".txt"]:
        pages = extract_text_file(file_path)
    else:
        return 0

    if not pages:
        console.print(f"[yellow]No text extracted from {file_path.name}[/yellow]")
        return 0

    console.print(f"  {len(pages)} page(s) extracted")

    chunks = chunk_pages(
        pages, source_name=source_name, game_system=game_system, source_path=source_path
    )
    console.print(f"  {len(chunks)} chunks created")

    if not chunks:
        return 0

    # Embed chunks in batches
    texts = [c.text for c in chunks]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding chunks...", total=len(texts))
        batch_size = EMBED_BATCH_SIZE
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embs = embed_texts(batch)
            all_embeddings.extend(embs)
            progress.update(task, advance=len(batch))

    # Store in ChromaDB
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = []
    for c in chunks:
        meta = {**c.metadata, "source_hash": file_hash}
        # ChromaDB requires string/int/float values
        for k, v in meta.items():
            if v is None:
                meta[k] = ""
        metadatas.append(meta)

    store.add(
        ids=ids,
        documents=texts,
        embeddings=all_embeddings,
        metadatas=metadatas,
    )

    console.print(f"  [green]Stored {len(chunks)} chunks[/green]")
    return len(chunks)


# Backward-compatible alias
ingest_pdf = ingest_file


def ingest_sourcebooks(path: str | None = None) -> None:
    """Scan sourcebooks directory (or specific path) and ingest new sourcebooks."""
    settings = get_settings()
    store = get_store()

    if path:
        target = Path(path)
        if target.is_file() and target.suffix.lower() in SUPPORTED_EXTENSIONS:
            source_files = [target]
        elif target.is_dir():
            source_files = sorted(
                p
                for p in target.rglob("*")
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
            )
        else:
            console.print(f"[red]Not a supported sourcebook file or directory: {target}[/red]")
            return
    else:
        sb_path = settings.sourcebooks_path
        if not sb_path.exists():
            console.print(f"[red]Sourcebooks directory not found: {sb_path}[/red]")
            console.print("Create it and add PDF/Markdown files, then run ingest again.")
            return
        source_files = sorted(
            p
            for p in sb_path.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    if not source_files:
        console.print("[yellow]No supported sourcebook files found.[/yellow]")
        return

    console.print(f"Found {len(source_files)} sourcebook file(s)\n")

    total_chunks = 0
    for file_path in source_files:
        total_chunks += ingest_file(file_path, store)

    console.print(f"\n[bold green]Done![/bold green] Total chunks in store: {store.count()}")
