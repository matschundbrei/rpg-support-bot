# RPG Support Bot

A RAG-powered assistant for tabletop RPG players and game masters. Ask questions about rules, world building, characters, and stories -- grounded in your own PDF source books with cited answers.

## Features

- **PDF ingestion pipeline** -- extracts text from RPG source books (multi-column, headings, stat blocks), chunks intelligently, and embeds into a local vector database
- **RAG retrieval** -- finds relevant passages from your source books and provides them as context to the LLM
- **Cited answers** -- every answer references `[Book Name, p.XX]` so you can verify
- **Multilingual** -- works with English and German source books; responds in the user's language
- **Game system filtering** -- organize books by system and filter queries accordingly
- **CLI and Web UI** -- interactive terminal chat or Gradio browser interface

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python package manager)
- A local embedding model served via [LM Studio](https://lmstudio.ai/), [Ollama](https://ollama.com/), or any OpenAI-compatible API (default model: `nomic-embed-text-v1.5`)
- One of the following for the LLM:
  - An [Anthropic API key](https://console.anthropic.com/) for Claude (default), **or**
  - A local model served via LM Studio, [Ollama](https://ollama.com/), or any OpenAI-compatible API

## Setup

```bash
# Clone and enter the project
cd rpg-support-bot

# Copy the environment file and add your API key (only needed for Anthropic backend)
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=sk-ant-...

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
```

## Running Tests

```bash
uv run pytest
```

## Architecture

```
User question
  -> Embed query (LM Studio API)
  -> Retrieve top-k chunks from ChromaDB
  -> Format as numbered context with source attribution
  -> Send to LLM (Claude API or local model) with system prompt + context
  -> Stream cited answer back to user
```

Both CLI and Web UI share the same retrieval and LLM backend. The LLM client supports Anthropic's API and any OpenAI-compatible endpoint (LM Studio, Ollama, etc.), switchable via `config.yaml`. Source books are stored in a single ChromaDB collection with metadata filtering for game system, language, page number, and section.
