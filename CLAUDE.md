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
uv run rpg-bot serve             # Launch web UI + API server on :8000
uv run rpg-bot list              # List ingested source books
```

## Architecture

The pipeline flows: **PDF extraction → chunking → embedding → ChromaDB storage → query-time retrieval → LLM with context**.

- **`config.py`** — Singleton `Settings` object built from `config.yaml` (tuning params) + `.env` (secrets). Uses pydantic-settings. Access via `get_settings()`.
- **`ingest/extract.py`** — PDF text extraction via PyMuPDF. Handles multi-column layouts and detects headings by font size heuristics. Returns `PageContent` per page.
- **`ingest/chunker.py`** — Recursive character splitter with overlap. Prepends breadcrumb context (`[Source > Section > Subsection]`) to each chunk. Detects language via langdetect.
- **`ingest/embed.py`** — Orchestrates the full ingest pipeline: extract → chunk → embed → store. Deduplicates by file SHA-256 hash. Embeds in batches of 16 (`EMBED_BATCH_SIZE`). Game system is derived from the subdirectory under the configured `sourcebooks_directory`.
- **`embeddings.py`** — Calls any OpenAI-compatible `/v1/embeddings` endpoint via the OpenAI SDK. Used by both ingest and query paths. Server URL configured via `embeddings.base_url`.
- **`retrieval/store.py`** — ChromaDB wrapper (`VectorStore`). Single collection with cosine similarity. Metadata includes source, page, game_system, language, section. Use `get_store()` (process-wide singleton) — opening a `PersistentClient` per call is expensive.
- **`retrieval/query.py`** — Hybrid RAG query: vector similarity + BM25 keyword ranking fused with Reciprocal Rank Fusion → deduplicate → threshold-filter → numbered context string. Returns `RAGResult` with context and a source_map for citation linking. BM25 index is cached at module level, keyed by store identity + chunk count.
- **`llm/client.py`** — Dual-backend streaming client (Anthropic SDK or OpenAI SDK). Backend selected via `llm.backend` config (`"anthropic"` or `"openai"`). Maintains a rolling conversation history (capped at `llm.max_history`) and rolls the user message back if the LLM call fails. Two system prompts: with and without RAG context.
- **`cli/chat.py`** — Interactive terminal chat using prompt_toolkit + rich. Renders streaming markdown, clickable citation links in terminal. `/sources` lists ingested books.
- **`api/server.py`** — FastAPI server exposing OpenAI-compatible `/v1/chat/completions` and `/v1/models` endpoints, plus `/api/chats` CRUD for chat persistence and `/api/game-systems`. Serves the web UI as static files at `/`. Each chat completion request runs through the RAG pipeline automatically. Optional bearer-token auth: set `API_KEY` in `.env` to protect `/v1/*` and `/api/*` (off by default). Game-system list and SDK clients are cached.
- **`persistence/database.py`** — SQLite with WAL mode, one connection per worker thread (FastAPI runs sync endpoints in a threadpool; connections are not shareable across threads). Schema: `chats` + `messages` tables. Database at `data/chats.db`. Messages order by `created_at, rowid` (timestamps are second-resolution).
- **`persistence/repository.py`** — CRUD functions for chats and messages. Auto-titles chats from first user message.
- **`static/`** — Single-page web UI built with Alpine.js, Tailwind CSS, marked.js, and DOMPurify (LLM/PDF-derived markdown is sanitized before rendering). Dark theme with streaming responses, game system filter, citation highlighting, and multi-chat sidebar. Prompts for the API key once when auth is enabled (stored in localStorage).

## Key Design Details

- **Embeddings are external**: Requires a running OpenAI-compatible embeddings server (LM Studio, Ollama, etc.) at the URL configured in `embeddings.base_url` (default `localhost:1234/v1`). No embedded model.
- **Game system filtering**: Derived from subdirectory name under `sourcebooks/` (e.g., `sourcebooks/dnd5e/` → `game_system="dnd5e"`). Applied as a ChromaDB `where` filter at query time.
- **Deduplication**: Ingested PDFs are tracked by SHA-256 hash in chunk metadata (`source_hash`). Re-running ingest skips already-processed files.
- **Citation linking**: The query layer builds `file://` URLs with `#page=N` anchors. CLI renders these as terminal hyperlinks; web UI highlights `[Book, p.XX]` citations as styled spans.
- **Chat persistence**: Web UI chats are stored in SQLite (`data/chats.db`). The `/v1/chat/completions` endpoint accepts an optional `chat_id` field for automatic message persistence. Chats are auto-titled from the first user message.
- **OpenAI-compatible API**: The `/v1/` endpoints allow external clients (Open WebUI, etc.) to use the bot. The `game_system` and `chat_id` fields are non-standard extensions ignored by generic clients.

## Configuration

- `config.example.yaml` — Template with all tunable parameters. Copy to `config.yaml` for local use (gitignored).
- `config.yaml` — Local config (not committed): LLM backend/model, embeddings server, chunk size/overlap, top_k, relevance threshold, ChromaDB path, etc.
- `.env` — `ANTHROPIC_API_KEY` and/or `OPENAI_API_KEY` depending on configured backends, plus optional `API_KEY` to protect the API server (loaded by pydantic-settings)
- Config classes in `config.py` merge YAML defaults with env var overrides
- **LLM backend**: Set `llm.backend` to `"anthropic"` (default) or `"openai"`. The `"openai"` backend works with the real OpenAI API (leave `base_url` empty, set `OPENAI_API_KEY`) or any compatible local server (set `base_url` to e.g. `http://localhost:1234/v1` for LM Studio, `http://localhost:11434/v1` for Ollama).
- **Embeddings server**: Set `embeddings.base_url` to any OpenAI-compatible `/v1` endpoint, or leave empty to use the real OpenAI API with `OPENAI_API_KEY`.

## Remote

- Hosted on GitHub: `https://github.com/matschundbrei/rpg-support-bot`

## Known Issues and Past Decisions

- **Python version**: The old ChromaDB 0.x line used Pydantic v1 and broke on Python 3.14. The current Rust-core ChromaDB 1.x (pinned `chromadb>=1.5.0`) works on 3.14 (verified with 1.5.9 on CPython 3.14.7). The project pins Python 3.14 via `.python-version` (moved up from 3.13 in September 2026 with full test suite + end-to-end verification).
- **PyMuPDF swigvarlink warning**: PyMuPDF's macOS wheels are built with SWIG 4.3.1 (SWIG 4.4.0 has a macOS bug, swig/swig#3279), and its generated types (SwigPyPacked, SwigPyObject, swigvarlink) lack `__module__`, so CPython emits harmless `DeprecationWarning`s when importing pymupdf. `ingest/extract.py` suppresses them right before the import — this covers the CLI/serve/ingest runtimes and the pytest warnings summary. One residual `<sys>:0: DeprecationWarning: builtin type swigvarlink...` line can still appear *after* the pytest result line: it is emitted at interpreter shutdown through CPython's C-level fallback once the `warnings` module itself is torn down, so no Python-level filter can catch it. All of this resolves itself upstream once PyMuPDF ships macOS wheels built with SWIG 4.4.0 (pymupdf/PyMuPDF#3931); drop the filter in `extract.py` when that happens. Do not "fix" it by promoting the warning to an error — that segfaults the interpreter during the SWIG extension import.
- **Embedding model choice**: Originally used local `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`) but retrieval quality for German RPG content was poor. Switched to a Nomic model served via an OpenAI-compatible API which dramatically improved results — removed ~2GB of torch/sentence-transformers dependencies in the process. Current default is `nomic-embed-text-v2-moe:latest` via Ollama. The store was re-ingested with this model (verified: re-embedding stored chunks with the configured model yields cosine ≈ 1.0, so the vectors are consistent with the current config).
- **Retrieval tuning history**: Default `chunk_size` was raised from 1000→1500 and `top_k` from 8→15, `relevance_threshold` relaxed from 0.3→1.0 (cosine distance). These values were tuned against German Shadowrun 6 sourcebooks. Smaller chunks and stricter thresholds caused relevant rules passages (e.g., Initiative on p.111) to not surface.
- **Breadcrumb noise**: Heading detection (`extract.py:_is_heading`) uses font-size heuristics that can be noisy on some PDFs. Breadcrumbs are limited to last 2 headings to reduce noise.
- **Scanned PDFs**: PDFs without extractable text (e.g., scanned cheat sheets) produce zero chunks silently. No OCR support currently.
- **Gradio removal**: The original Gradio web UI was replaced with a custom Alpine.js + Tailwind CSS frontend served by FastAPI. Voice input (STT/whisper.cpp) was removed at the same time.
