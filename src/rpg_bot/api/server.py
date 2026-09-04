"""OpenAI-compatible API server that wraps the RAG pipeline."""

from __future__ import annotations

import hmac
import json
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rpg_bot.config import get_settings
from rpg_bot.llm.prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_WITH_RAG
from rpg_bot.persistence import repository as repo
from rpg_bot.retrieval.query import query_rag
from rpg_bot.retrieval.store import get_store

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

# Paths that never require authentication (UI shell + static assets)
_PUBLIC_PREFIXES = ("/", "/static")


def _check_auth(request: Request) -> bool:
    """Return True if the request is allowed. Auth is only enforced when
    API_KEY is set in .env; otherwise the server stays open (backward compat)."""
    key = get_settings().api_key
    if not key:
        return True
    path = request.url.path
    if path in _PUBLIC_PREFIXES or path.startswith("/static") or path in (
        "/docs",
        "/redoc",
        "/openapi.json",
    ):
        return True
    auth = request.headers.get("authorization", "")
    expected = f"Bearer {key}"
    return hmac.compare_digest(auth, expected)


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if not _check_auth(request):
        return JSONResponse(
            status_code=401,
            content={"error": "Invalid or missing API key. Send: Authorization: Bearer <key>"},
        )
    return await call_next(request)


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


# --- Cached lookups ---

_game_systems_cache: tuple[int, list[str]] | None = None


def _get_game_systems() -> list[str]:
    """Extract unique game system identifiers from ChromaDB metadata.

    Cached and invalidated when the chunk count changes (i.e. after ingest/delete).
    """
    global _game_systems_cache
    store = get_store()
    count = store.count()
    if _game_systems_cache is not None and _game_systems_cache[0] == count:
        return _game_systems_cache[1]
    try:
        all_meta = store.collection.get(include=["metadatas"])
        systems = set()
        for meta in all_meta["metadatas"] or []:
            gs = meta.get("game_system")
            if gs:
                systems.add(gs)
        _game_systems_cache = (count, sorted(systems))
        return _game_systems_cache[1]
    except Exception:
        return []


_llm_clients: dict[tuple, object] = {}


def _get_llm_client(backend: str, base_url: str, api_key: str):
    """Cache SDK clients per (backend, base_url, key) — they are thread-safe."""
    key = (backend, base_url, api_key)
    if key not in _llm_clients:
        if backend == "anthropic":
            import anthropic

            _llm_clients[key] = anthropic.Anthropic(api_key=api_key)
        else:
            from openai import OpenAI

            kwargs = {}
            if base_url:
                kwargs["base_url"] = base_url
            _llm_clients[key] = OpenAI(
                api_key=api_key or "no-key-required",
                **kwargs,
            )
    return _llm_clients[key]


def _call_llm_stream(
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
):
    """Call the configured LLM backend and yield text chunks."""
    settings = get_settings()
    backend = settings.llm.backend

    if backend == "anthropic":
        client = _get_llm_client(backend, "", settings.anthropic_api_key)
        system = messages[0]["content"]
        chat_messages = messages[1:]
        with client.messages.stream(
            model=settings.llm.model,
            max_tokens=max_tokens,
            system=system,
            messages=chat_messages,
            extra_body={"temperature": temperature},
        ) as stream:
            for text in stream.text_stream:
                yield text
    else:
        client = _get_llm_client(
            backend, settings.llm.base_url, settings.openai_api_key
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


def _make_chunk_bytes(
    chunk_id: str, content: str, finish: bool = False, role: str | None = None
) -> bytes:
    """Format a single SSE chunk in OpenAI format."""
    if finish:
        delta: dict = {}
    elif role is not None:
        delta = {"role": role, "content": content}
    else:
        delta = {"content": content}
    data = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "delta": delta,
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

    # Validate chat_id before doing any work
    if request.chat_id and not repo.chat_exists(request.chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")

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
            # First chunk carries the assistant role, per the OpenAI protocol
            yield _make_chunk_bytes(chunk_id, "", role="assistant")
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

    # Rough token estimates (~4 chars/token) so clients get sane stats
    prompt_chars = sum(len(m["content"]) for m in messages)
    completion_tokens = max(1, len(full) // 4)

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
        "usage": {
            "prompt_tokens": max(1, prompt_chars // 4),
            "completion_tokens": completion_tokens,
            "total_tokens": max(1, prompt_chars // 4) + completion_tokens,
        },
    }


# Mount static files AFTER all routes so they don't shadow API endpoints
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
