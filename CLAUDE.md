# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG-powered assistant for tabletop RPG players and GMs. Ingests PDF source books into a local ChromaDB vector store, retrieves relevant chunks at query time, and sends them as context to an LLM for cited answers. Supports English and German source books. Both the LLM and embedding backends are swappable between Anthropic Claude and any OpenAI-compatible API (LM Studio, Ollama, etc.).

## Commands

```bash
uv sync                          # Install dependencies
uv run pytest                    # Run all tests
uv run pytest tests/test_chunker.py  # Run a single test file
uv run pytest -k test_split_text_short  # Run a single test

uv run rpg-bot ingest            # Ingest all PDFs from sourcebooks/
uv run rpg-bot chat              # CLI chat
uv run rpg-bot chat -s dnd5e     # CLI chat filtered to a game system
uv run rpg-bot web               # Launch Gradio web UI on :7860
uv run rpg-bot list              # List ingested source books
```

## Architecture

The pipeline flows: **PDF extraction → chunking → embedding → ChromaDB storage → query-time retrieval → LLM with context**.

- **`config.py`** — Singleton `Settings` object built from `config.yaml` (tuning params) + `.env` (secrets). Uses pydantic-settings. Access via `get_settings()`.
- **`ingest/extract.py`** — PDF text extraction via PyMuPDF. Handles multi-column layouts and detects headings by font size heuristics. Returns `PageContent` per page.
- **`ingest/chunker.py`** — Recursive character splitter with overlap. Prepends breadcrumb context (`[Source > Section > Subsection]`) to each chunk. Detects language via langdetect.
- **`ingest/embed.py`** — Orchestrates the full ingest pipeline: extract → chunk → embed → store. Deduplicates by file SHA-256 hash. Embeds in batches of 64.
- **`embeddings.py`** — Calls any OpenAI-compatible `/v1/embeddings` endpoint via the OpenAI SDK. Used by both ingest and query paths. Server URL configured via `embeddings.base_url`.
- **`retrieval/store.py`** — ChromaDB wrapper (`VectorStore`). Single collection with cosine similarity. Metadata includes source, page, game_system, language, section.
- **`retrieval/query.py`** — RAG query: embed user question → retrieve top-k chunks → filter by distance threshold → deduplicate → format numbered context string. Returns `RAGResult` with context and a source_map for citation linking.
- **`llm/client.py`** — Dual-backend streaming client (Anthropic SDK or OpenAI SDK). Backend selected via `llm.backend` config (`"anthropic"` or `"openai"`). Maintains conversation history. Two system prompts: with and without RAG context.
- **`cli/chat.py`** — Interactive terminal chat using prompt_toolkit + rich. Renders streaming markdown, clickable citation links in terminal.
- **`web/app.py`** — Gradio Blocks UI with game system dropdown. Creates a new `LLMClient` per request, reconstructing history from Gradio's message list.

## Key Design Details

- **Embeddings are external**: Requires a running OpenAI-compatible embeddings server (LM Studio, Ollama, etc.) at the URL configured in `embeddings.base_url` (default `localhost:1234/v1`). No embedded model.
- **Game system filtering**: Derived from subdirectory name under `sourcebooks/` (e.g., `sourcebooks/dnd5e/` → `game_system="dnd5e"`). Applied as a ChromaDB `where` filter at query time.
- **Deduplication**: Ingested PDFs are tracked by SHA-256 hash in chunk metadata (`source_hash`). Re-running ingest skips already-processed files.
- **Citation linking**: The query layer builds `file://` URLs with `#page=N` anchors. CLI renders these as terminal hyperlinks; web UI converts them to markdown links.

## Configuration

- `config.yaml` — All tunable parameters: LLM backend/model, embeddings server, chunk size/overlap, top_k, relevance threshold, ChromaDB path, etc.
- `.env` — `ANTHROPIC_API_KEY` and/or `OPENAI_API_KEY` depending on configured backends (loaded by pydantic-settings)
- Config classes in `config.py` merge YAML defaults with env var overrides
- **LLM backend**: Set `llm.backend` to `"anthropic"` (default) or `"openai"`. The `"openai"` backend works with the real OpenAI API (leave `base_url` empty, set `OPENAI_API_KEY`) or any compatible local server (set `base_url` to e.g. `http://localhost:1234/v1` for LM Studio, `http://localhost:11434/v1` for Ollama).
- **Embeddings server**: Set `embeddings.base_url` to any OpenAI-compatible `/v1` endpoint, or leave empty to use the real OpenAI API with `OPENAI_API_KEY`.

## Remote

- Hosted on Codeberg: `ssh://git@codeberg.org/maub/rpg-bot.git`

## Known Issues and Past Decisions

- **Python 3.14 incompatible**: ChromaDB internally uses Pydantic v1 which breaks on Python 3.14. Project uses Python 3.13 (set via `uv python pin`). Don't attempt upgrading until ChromaDB fixes this upstream.
- **Embedding model choice**: Originally used local `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`) but retrieval quality for German RPG content was poor. Switched to `nomic-embed-text-v1.5` served via OpenAI-compatible API which dramatically improved results — removed ~2GB of torch/sentence-transformers dependencies in the process.
- **Retrieval tuning history**: Default `chunk_size` was raised from 1000→1500 and `top_k` from 8→15, `relevance_threshold` relaxed from 0.3→1.0 (cosine distance). These values were tuned against German Shadowrun 6 sourcebooks. Smaller chunks and stricter thresholds caused relevant rules passages (e.g., Initiative on p.111) to not surface.
- **Breadcrumb noise**: Heading detection (`extract.py:_is_heading`) uses font-size heuristics that can be noisy on some PDFs. Breadcrumbs are limited to last 2 headings to reduce noise.
- **Gradio web UI**: Required explicit `server_port` in `launch()` to avoid port-scanning errors. The Gradio 5→6 migration needed `gr.ChatInterface` with `type="messages"` (handled in a prior fix).
- **Scanned PDFs**: PDFs without extractable text (e.g., scanned cheat sheets) produce zero chunks silently. No OCR support currently.
