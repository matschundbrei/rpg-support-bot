# RPG Support Bot

A RAG-powered assistant for tabletop RPG players and game masters. Ask questions about rules, world building, characters, and stories -- grounded in your own PDF source books with cited answers.

## Features

- **PDF ingestion pipeline** -- extracts text from RPG source books (multi-column, headings, stat blocks), chunks intelligently, and embeds into a local vector database
- **RAG retrieval** -- finds relevant passages from your source books and provides them as context to the LLM
- **Cited answers** -- every answer references `[Book Name, p.XX]` so you can verify
- **Multilingual** -- works with English and German source books; responds in the user's language
- **Game system filtering** -- organize books by system and filter queries accordingly
- **CLI and Web UI** -- interactive terminal chat or Gradio browser interface
- **Voice input** -- optional speech-to-text via [whisper.cpp](https://github.com/ggerganov/whisper.cpp) server (works in all browsers)

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python package manager)
- A local embedding model served via [LM Studio](https://lmstudio.ai/), [Ollama](https://ollama.com/), or any OpenAI-compatible API (default model: `nomic-embed-text-v1.5`)
- One of the following for the LLM:
  - An [Anthropic API key](https://console.anthropic.com/) for Claude (default), **or**
  - An [OpenAI API key](https://platform.openai.com/api-keys) for GPT models, **or**
  - A local model served via LM Studio, [Ollama](https://ollama.com/), or any OpenAI-compatible API

## Setup

```bash
# Clone and enter the project
git clone https://codeberg.org/maub/rpg-bot.git
cd rpg-bot

# Copy the environment file and add your API key(s)
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY and/or OPENAI_API_KEY

# Install dependencies
uv sync
```

### Embedding Model

Start LM Studio (or Ollama) and load an embedding model. The default config expects `text-embedding-nomic-embed-text-v1.5` on `http://localhost:1234`. Any server with an OpenAI-compatible `/v1/embeddings` endpoint works:

```yaml
embeddings:
  model: "text-embedding-nomic-embed-text-v1.5"
  base_url: "http://localhost:1234/v1"     # LM Studio
  # base_url: "http://localhost:11434/v1"  # Ollama
```

For Ollama, pull the model first: `ollama pull nomic-embed-text`

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
- `/sources` -- show active filter
- `/clear` -- reset conversation history
- `/quit` -- exit

### Web UI

```bash
uv run rpg-bot web
```

Opens a Gradio chat interface at `http://localhost:7860` with a game system dropdown.

#### Voice input (optional)

The web UI supports voice input via a microphone recording button. Audio is transcribed locally using a [whisper.cpp](https://github.com/ggerganov/whisper.cpp) server. To enable it:

1. **Install whisper.cpp** and download a model:

   ```bash
   # Build from source (macOS with Metal acceleration)
   git clone https://github.com/ggerganov/whisper.cpp.git
   cd whisper.cpp
   cmake -B build
   cmake --build build --config Release

   # Download a model (medium is a good balance for English + German)
   ./models/download-ggml-model.sh medium
   ```

2. **Start the whisper.cpp server:**

   ```bash
   ./build/bin/whisper-server -m models/ggml-medium.bin -l auto --convert
   ```

   - `-l auto` detects the spoken language automatically
   - `--convert` accepts non-WAV audio formats (requires ffmpeg)
   - Server runs on `http://localhost:8080` by default

3. **Enable STT** in `config.yaml`:

   ```yaml
   stt:
     enabled: true
   ```

In the web UI, a microphone widget will appear. Record your question, then click Send.

### List ingested books

```bash
uv run rpg-bot list
```

## Configuration

All settings are in `config.yaml`:

```yaml
llm:
  backend: "anthropic"           # "anthropic" or "openai"
  model: "claude-sonnet-4-20250514"
  max_tokens: 4096
  temperature: 0.3
  # base_url: "http://localhost:1234/v1"  # only used with "openai" backend

embeddings:
  model: "text-embedding-nomic-embed-text-v1.5"
  base_url: "http://localhost:1234/v1"

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

### Speech-to-text

Voice input defaults to a local whisper.cpp server. You can also use any OpenAI-compatible transcription endpoint:

```yaml
stt:
  enabled: true
  backend: "whisper-cpp"                   # "whisper-cpp" or "openai"
  base_url: "http://localhost:8080"        # whisper.cpp server default

  # For OpenAI-compatible backends (faster-whisper-server, Ollama, OpenAI API),
  # set backend: "openai" and adjust base_url:
  # backend: "openai"
  # base_url: ""                           # real OpenAI API (needs OPENAI_API_KEY)
  # base_url: "http://localhost:8080/v1"   # faster-whisper-server
```

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
  -> Embed chunks via OpenAI-compatible API (batches of 64)
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

Both CLI and Web UI share the same retrieval and LLM backend. Source books are stored in a single ChromaDB collection with metadata filtering for game system, language, page number, and section.
