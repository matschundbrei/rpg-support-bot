# RPG Support Bot

A RAG-powered assistant for tabletop RPG players and game masters. Ask questions about rules, world building, characters, and stories -- grounded in your own PDF source books with cited answers.

## Features

- **PDF ingestion pipeline** -- extracts text from RPG source books (multi-column, headings, stat blocks), chunks intelligently, and embeds into a local vector database
- **RAG retrieval** -- finds relevant passages from your source books and provides them as context to the LLM
- **Cited answers** -- every answer references `[Book Name, p.XX]` so you can verify
- **Multilingual** -- works with English and German source books; responds in the user's language
- **Game system filtering** -- organize books by system and filter queries accordingly
- **CLI and Web UI** -- interactive terminal chat or browser-based chat interface with multi-chat management
- **OpenAI-compatible API** -- use with Open WebUI or any OpenAI-compatible client for chat persistence and management

## Vibe Code Warning

This repository's contents are "vibe coded" with several different coding harnesses.

I have previously attempted to do this on my own, with little success.

My 'wasted hours' counter on solo attempts had hit about 60 before I dropped the project during my last vacation. This version — built with the coding harnesses — took me about 6 to 10 hours and approximately 400k tokens.

And I got somewhere. The result is clearly not perfect, but it's a lot more "works for me" than what I had before.

I will extend this codebase and continuously review and upgrade it while testing new models and harnesses. Also feel free to fork or contribute via PRs.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python package manager)
- A local embedding model served via [LM Studio](https://lmstudio.ai/), [Ollama](https://ollama.com/), or any OpenAI-compatible API (default model: `nomic-embed-text-v2-moe:latest`)
- One of the following for the LLM:
  - An [Anthropic API key](https://console.anthropic.com/) for Claude (default), **or**
  - An [OpenAI API key](https://platform.openai.com/api-keys) for GPT models, **or**
  - A local model served via LM Studio, [Ollama](https://ollama.com/), or any OpenAI-compatible API

## Setup

```bash
# Clone and enter the project
git clone https://github.com/matschundbrei/rpg-support-bot.git
cd rpg-support-bot

# Copy config files and add your API key(s)
cp .env.example .env
cp config.example.yaml config.yaml
# Edit .env and set ANTHROPIC_API_KEY and/or OPENAI_API_KEY
# Edit config.yaml to configure LLM backend, embeddings server, etc.

# Install dependencies
uv sync
```

### Embedding Model

Start LM Studio (or Ollama) and load an embedding model. The default config expects `nomic-embed-text-v2-moe:latest` on Ollama at `http://localhost:11434`. Any server with an OpenAI-compatible `/v1/embeddings` endpoint works:

```yaml
embeddings:
  model: "nomic-embed-text-v2-moe:latest"
  base_url: "http://localhost:11434/v1"     # Ollama
  # base_url: "http://localhost:1234/v1"    # LM Studio
```

> **Note:** The embedding model must match the one used to ingest your existing corpus, otherwise similarity scores become meaningless. If you switch models, re-run `rpg-bot ingest` after clearing the store.

For Ollama, pull the model first: `ollama pull nomic-embed-text-v2-moe`

## Adding Source Books

Drop PDF files into the `sourcebooks/` directory. Organize by game system using subdirectories:

```
sourcebooks/
  dnd5e/
    players-handbook.pdf
    dungeon-masters-guide.pdf
  shadowrun6/
    grundregelwerk.pdf
    feuer-frei.pdf
```

The subdirectory name becomes the game system filter. PDFs placed directly in `sourcebooks/` have no system tag.

> **Note:** PDF files are gitignored and will not be committed to the repository.

## Usage

### Ingest source books

```bash
# Ingest all PDFs from sourcebooks/
uv run rpg-bot ingest

# Ingest a specific file or directory
uv run rpg-bot ingest -p sourcebooks/shadowrun6/grundregelwerk.pdf
uv run rpg-bot ingest -p sourcebooks/dnd5e/
```

### Interactive chat (CLI)

```bash
# Start chatting (uses all ingested books)
uv run rpg-bot chat

# Filter by game system
uv run rpg-bot chat -s shadowrun6
```

In-chat commands:
- `/system <name>` -- set game system filter (e.g. `/system dnd5e`)
- `/system` -- clear the filter
- `/sources` -- list ingested sources and the active filter
- `/clear` -- reset conversation history
- `/quit` -- exit

### Web UI

```bash
uv run rpg-bot serve
```

Opens the web chat interface at `http://localhost:8000` with:
- Multi-chat sidebar (create, rename, delete chats)
- Game system filter per chat
- Streaming responses with Markdown rendering
- Citation highlighting (`[Book Name, p.XX]`)
- Chat persistence in SQLite

Use `--port` to change the port, e.g. `uv run rpg-bot serve --port 3000`.

#### Using with Open WebUI or other clients

The same server exposes an OpenAI-compatible API at `http://localhost:8000/v1`. Point any compatible client at this URL to use the RAG pipeline with your own chat frontend.

#### Authentication (optional)

By default the server is open on localhost. If you bind it to a network interface (`--host 0.0.0.0`) or expose it otherwise, set `API_KEY` in `.env` to require a bearer token:

```
API_KEY=some-long-random-string
```

With `API_KEY` set, all `/v1/*` and `/api/*` requests must send `Authorization: Bearer <API_KEY>`. The built-in web UI keeps working: it prompts once for the key and stores it in the browser's localStorage.

### List ingested books

```bash
uv run rpg-bot list
```

## Configuration

Copy `config.example.yaml` to `config.yaml` and adjust as needed:

```yaml
llm:
  backend: "anthropic"           # "anthropic" or "openai"
  model: "claude-sonnet-4-6"
  max_tokens: 4096
  temperature: 0.3
  # base_url: "http://localhost:1234/v1"  # only used with "openai" backend

embeddings:
  model: "nomic-embed-text-v2-moe:latest"
  base_url: "http://localhost:11434/v1"    # Ollama

chunking:
  chunk_size: 1500
  chunk_overlap: 200

retrieval:
  top_k: 15
  relevance_threshold: 1.0

chromadb:
  persist_directory: "data/chromadb"
  collection_name: "rpg_sourcebooks"
```

### Using the OpenAI API

Set `backend: "openai"` and add your key to `.env`. Leave `base_url` empty to use the OpenAI default API (api.openai.com/v1):

```yaml
llm:
  backend: "openai"
  model: "gpt-4o"
  max_tokens: 4096
  temperature: 0.3
```

You can also use OpenAI for embeddings by setting `embeddings.base_url` to empty and choosing a model like `text-embedding-3-small`.

### Using a local LLM

Set `backend: "openai"` and point `base_url` at your local server:

```yaml
llm:
  backend: "openai"
  model: "qwen3-32b"                      # model name as shown in your server
  max_tokens: 4096
  temperature: 0.3
  base_url: "http://localhost:1234/v1"     # LM Studio
  # base_url: "http://localhost:11434/v1"  # Ollama
```

No API key is needed for local models.

### Secrets

Secrets go in `.env` (not committed):

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

## Running Tests

```bash
uv run pytest
```

## Architecture

### Pipeline

```
PDF source books
  -> Extract text per page (PyMuPDF, multi-column aware)
  -> Detect headings by font size heuristics
  -> Split pages into sections at heading boundaries
  -> Chunk sections with recursive character splitter + overlap
  -> Prepend breadcrumb context ([Source > Section]) to each chunk
  -> Embed chunks via OpenAI-compatible API (batches of 16)
  -> Store in ChromaDB with metadata (source, page, game system, language, section)
```

```
User question
  -> Embed query via OpenAI-compatible API
  -> Vector similarity search (top-k × 3 candidates)
  -> BM25 keyword search over full corpus (top-k × 3 candidates)
  -> Reciprocal Rank Fusion to merge both rankings
  -> Deduplicate and threshold-filter to final top-k
  -> Format as numbered context with source attribution
  -> Send to LLM (Anthropic or OpenAI-compatible) with system prompt + context
  -> Stream cited answer back to user
```

### Key components

- **PDF extraction** (`ingest/extract.py`) -- uses `get_text("text")` for content (handles multi-column layouts correctly) and `get_text("dict")` only for heading detection. Deduplicates layered PDF text.
- **Section-aware chunking** (`ingest/chunker.py`) -- splits page text at detected heading boundaries so each spell, item, or rule gets its own chunk. Small sections (<150 chars) merge into their predecessor.
- **Hybrid retrieval** (`retrieval/query.py`) -- combines vector similarity with BM25 keyword matching via Reciprocal Rank Fusion (RRF). Keyword matches bypass the distance threshold, so exact term matches always surface.
- **Dual LLM backend** (`llm/client.py`) -- Anthropic SDK or OpenAI SDK, switchable via `config.yaml`. Supports streaming.
- **Embeddings** (`embeddings.py`) -- calls any OpenAI-compatible `/v1/embeddings` endpoint. Works with LM Studio, Ollama, or the real OpenAI API.

- **Web UI** (`static/`) -- Single-page app built with Alpine.js, Tailwind CSS, and marked.js. Served by FastAPI as static files. Multi-chat management with SQLite persistence.
- **API server** (`api/server.py`) -- OpenAI-compatible `/v1/chat/completions` endpoint that transparently runs RAG retrieval. Also serves chat CRUD endpoints under `/api/` for the web UI.
- **Chat persistence** (`persistence/`) -- SQLite database (`data/chats.db`) for storing chats and messages. Auto-titles chats from the first user message.

Both CLI and Web UI share the same retrieval and LLM backend. Source books are stored in a single ChromaDB collection with metadata filtering for game system, language, page number, and section.
