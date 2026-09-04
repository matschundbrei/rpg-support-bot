from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from rpg_bot.config import get_settings
from rpg_bot.embeddings import embed_texts
from rpg_bot.ingest.chunker import chunk_pages
from rpg_bot.ingest.extract import extract_pdf
from rpg_bot.retrieval.store import VectorStore, get_store

console = Console()

EMBED_BATCH_SIZE = 16


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


def ingest_pdf(pdf_path: Path, store: VectorStore) -> int:
    """Ingest a single PDF. Returns number of chunks added."""
    file_hash = _file_hash(pdf_path)

    if store.has_source(file_hash):
        console.print(f"[dim]Skipping (already ingested): {pdf_path.name}[/dim]")
        return 0

    source_name = pdf_path.stem
    source_path = str(pdf_path.resolve())
    game_system = _guess_game_system(pdf_path)

    console.print(f"[bold]Extracting:[/bold] {pdf_path.name}")
    pages = extract_pdf(pdf_path)
    if not pages:
        console.print(f"[yellow]No text extracted from {pdf_path.name}[/yellow]")
        return 0

    console.print(f"  {len(pages)} pages extracted")

    chunks = chunk_pages(
        pages, source_name=source_name, game_system=game_system, source_path=source_path
    )
    console.print(f"  {len(chunks)} chunks created")

    if not chunks:
        return 0

    # Embed chunks in batches via LM Studio API
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


def ingest_sourcebooks(path: str | None = None) -> None:
    """Scan sourcebooks directory (or specific path) and ingest new PDFs."""
    settings = get_settings()
    store = get_store()

    if path:
        target = Path(path)
        if target.is_file() and target.suffix.lower() == ".pdf":
            pdf_files = [target]
        elif target.is_dir():
            pdf_files = sorted(target.rglob("*.pdf"))
        else:
            console.print(f"[red]Not a PDF file or directory: {target}[/red]")
            return
    else:
        sb_path = settings.sourcebooks_path
        if not sb_path.exists():
            console.print(f"[red]Sourcebooks directory not found: {sb_path}[/red]")
            console.print("Create it and add PDF files, then run ingest again.")
            return
        pdf_files = sorted(sb_path.rglob("*.pdf"))

    if not pdf_files:
        console.print("[yellow]No PDF files found.[/yellow]")
        return

    console.print(f"Found {len(pdf_files)} PDF file(s)\n")

    total_chunks = 0
    for pdf_path in pdf_files:
        total_chunks += ingest_pdf(pdf_path, store)

    console.print(f"\n[bold green]Done![/bold green] Total chunks in store: {store.count()}")
