"""OpenAI-compatible API server that wraps the RAG pipeline."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rpg_bot.config import get_settings
from rpg_bot.llm.prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_WITH_RAG
from rpg_bot.persistence import repository as repo
from rpg_bot.retrieval.query import query_rag
from rpg_bot.retrieval.store import VectorStore

app = FastAPI(title="rpg-bot API")


def configure_cors(origins: list[str]) -> None:
    """Enable CORS with the given origin list. Call before server starts."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )


MODEL_ID = "rpg-bot"
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


# --- Request / Response schemas (OpenAI-compatible subset) ---


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    game_system: str | None = Field(
        None,
        description="Optional RPG game system filter (e.g. 'dnd5e', 'sr6'). "
        "Non-standard field, ignored by generic clients.",
    )
    chat_id: str | None = Field(
        None,
        description="Optional chat ID for persistence. "
        "Non-standard field, used by the built-in web UI.",
    )


class CreateChatRequest(BaseModel):
    game_system: str | None = None


class UpdateChatRequest(BaseModel):
    title: str | None = None
    game_system: str | None = None


# --- Helpers ---


def _extract_user_query(messages: list[ChatMessage]) -> str:
    """Return the last user message content."""
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content
    return messages[-1].content if messages else ""


def _build_messages(
    request: ChatCompletionRequest,
    system_prompt: str,
) -> list[dict[str, str]]:
    """Build the message list for the LLM, inserting the system prompt."""
    oai_messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
    ]
    for msg in request.messages:
        if msg.role == "system":
            continue
        oai_messages.append({"role": msg.role, "content": msg.content})
    return oai_messages


def _get_system_prompt(context: str | None) -> str:
    if context:
        return SYSTEM_PROMPT_WITH_RAG.format(context=context)
    return SYSTEM_PROMPT


def _get_game_systems() -> list[str]:
    """Extract unique game system identifiers from ChromaDB metadata."""
    try:
        store = VectorStore()
        all_meta = store.collection.get(include=["metadatas"])
        systems = set()
        for meta in all_meta["metadatas"] or []:
            gs = meta.get("game_system")
            if gs:
                systems.add(gs)
        return sorted(systems)
    except Exception:
        return []


def _call_llm_stream(
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
):
    """Call the configured LLM backend and yield text chunks."""
    settings = get_settings()
    backend = settings.llm.backend

    if backend == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        system = messages[0]["content"]
        chat_messages = messages[1:]
        with client.messages.stream(
            model=settings.llm.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=chat_messages,
        ) as stream:
            for text in stream.text_stream:
                yield text
    else:
        from openai import OpenAI

        kwargs = {}
        if settings.llm.base_url:
            kwargs["base_url"] = settings.llm.base_url
        client = OpenAI(
            api_key=settings.openai_api_key or "no-key-required",
            **kwargs,
        )
        stream = client.chat.completions.create(
            model=settings.llm.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content


def _make_chunk_bytes(chunk_id: str, content: str, finish: bool = False) -> bytes:
    """Format a single SSE chunk in OpenAI format."""
    data = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "delta": {} if finish else {"content": content},
                "finish_reason": "stop" if finish else None,
            }
        ],
    }
    return f"data: {json.dumps(data)}\n\n".encode()


# --- Web UI ---


@app.get("/", include_in_schema=False)
def serve_index():
    return FileResponse(STATIC_DIR / "index.html")


# --- Game Systems ---


@app.get("/api/game-systems")
def get_game_systems():
    return {"game_systems": _get_game_systems()}


# --- Chat CRUD ---


@app.get("/api/chats")
def list_chats():
    return repo.list_chats()


@app.post("/api/chats", status_code=201)
def create_chat(request: CreateChatRequest):
    return repo.create_chat(game_system=request.game_system)


@app.get("/api/chats/{chat_id}")
def get_chat(chat_id: str):
    chat = repo.get_chat(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@app.put("/api/chats/{chat_id}")
def update_chat(chat_id: str, request: UpdateChatRequest):
    chat = repo.update_chat(chat_id, title=request.title, game_system=request.game_system)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@app.delete("/api/chats/{chat_id}", status_code=204)
def delete_chat(chat_id: str):
    if not repo.delete_chat(chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")


# --- OpenAI-compatible endpoints ---


@app.get("/v1/models")
def list_models():
    """Return available models so Open WebUI can discover us."""
    game_systems = _get_game_systems()
    models = [
        {
            "id": MODEL_ID,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "rpg-bot",
            "permission": [],
            "root": MODEL_ID,
            "parent": None,
        }
    ]
    return {"object": "list", "data": models}


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    settings = get_settings()
    temperature = request.temperature if request.temperature is not None else settings.llm.temperature
    max_tokens = request.max_tokens if request.max_tokens is not None else settings.llm.max_tokens

    # RAG retrieval
    user_query = _extract_user_query(request.messages)
    rag_result = query_rag(user_query, game_system=request.game_system)
    context = rag_result.context if rag_result else None
    system_prompt = _get_system_prompt(context)
    messages = _build_messages(request, system_prompt)

    # Persist user message if chat_id provided
    if request.chat_id:
        repo.add_message(request.chat_id, "user", user_query)
        repo.auto_title(request.chat_id, user_query)

    if request.stream:
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        chat_id_for_persist = request.chat_id

        def event_stream():
            full_response = ""
            for text in _call_llm_stream(messages, temperature, max_tokens):
                full_response += text
                yield _make_chunk_bytes(chunk_id, text)
            yield _make_chunk_bytes(chunk_id, "", finish=True)
            yield b"data: [DONE]\n\n"
            # Persist assistant response after streaming completes
            if chat_id_for_persist:
                repo.add_message(chat_id_for_persist, "assistant", full_response)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
        )

    # Non-streaming: collect full response
    full = "".join(_call_llm_stream(messages, temperature, max_tokens))

    # Persist assistant response
    if request.chat_id:
        repo.add_message(request.chat_id, "assistant", full)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": full},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


# Mount static files AFTER all routes so they don't shadow API endpoints
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
