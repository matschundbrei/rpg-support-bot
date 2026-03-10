from __future__ import annotations

import logging
import re
from collections.abc import Generator

import gradio as gr

from rpg_bot.config import get_settings
from rpg_bot.llm.client import LLMClient
from rpg_bot.retrieval.query import query_rag
from rpg_bot.retrieval.store import VectorStore

logger = logging.getLogger(__name__)


def _get_game_systems() -> list[str]:
    try:
        store = VectorStore()
        results = store.collection.get(include=["metadatas"], limit=10000)
        systems = set()
        for meta in results.get("metadatas", []):
            if meta and meta.get("game_system"):
                systems.add(meta["game_system"])
        return ["(all)"] + sorted(systems)
    except Exception:
        return ["(all)"]


def _get_source_paths() -> list[str]:
    """Get all unique source PDF paths for Gradio's allowed_paths."""
    try:
        store = VectorStore()
        results = store.collection.get(include=["metadatas"], limit=10000)
        paths = set()
        for meta in results.get("metadatas", []):
            if meta and meta.get("source_path"):
                paths.add(meta["source_path"])
        return sorted(paths)
    except Exception:
        return []


_CITATION_RE = re.compile(r"\[([^]]+?,\s*p\.\s*(\d+))\]")


def _linkify_citations(text: str, source_map: dict[str, str]) -> str:
    """Replace [Book, p.XX] citations with markdown links to the PDF."""
    if not source_map:
        return text

    def _replace(match: re.Match) -> str:
        citation_key = match.group(1)
        url = source_map.get(citation_key, "")
        if url:
            return f"[{citation_key}]({url})"
        return match.group(0)

    return _CITATION_RE.sub(_replace, text)


def _transcribe_audio(audio_path: str) -> str:
    """Transcribe audio file using an OpenAI-compatible Whisper endpoint."""
    settings = get_settings()
    from openai import OpenAI

    client_kwargs: dict = {"api_key": settings.openai_api_key or "not-needed"}
    if settings.stt.base_url:
        client_kwargs["base_url"] = settings.stt.base_url
    client = OpenAI(**client_kwargs)

    with open(audio_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model=settings.stt.model,
            file=f,
        )
    return transcript.text


def _chat_response(
    message: dict,
    history: list[dict[str, str]],
    game_system: str,
    audio: str | None = None,
) -> Generator[str, None, None]:
    user_text = message.get("text", "") if isinstance(message, dict) else str(message)

    # Transcribe audio input if provided and no text was typed
    if audio and not user_text.strip():
        try:
            user_text = _transcribe_audio(audio)
        except Exception:
            logger.exception("Speech-to-text transcription failed")
            yield "⚠ Speech-to-text transcription failed. Check your STT configuration."
            return

    llm = LLMClient()

    # Restore conversation history
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ("user", "assistant") and content:
            llm.history.append({"role": role, "content": content})

    # RAG retrieval
    system_filter = game_system if game_system != "(all)" else None
    context = None
    source_map: dict[str, str] = {}
    try:
        rag_result = query_rag(user_text, game_system=system_filter)
        if rag_result is not None:
            context = rag_result.context
            source_map = rag_result.source_map
    except Exception:
        pass

    # Stream response, linkify citations in final output
    collected = ""
    for chunk in llm.chat_stream(user_text, context=context):
        collected += chunk
        yield _linkify_citations(collected, source_map)


def launch_app() -> None:
    settings = get_settings()
    game_systems = _get_game_systems()
    allowed_paths = _get_source_paths()

    stt_enabled = settings.stt.enabled

    with gr.Blocks(title="RPG Support Bot") as demo:
        gr.Markdown("# RPG Support Bot\nAsk questions about RPG rules, world building, characters, and more.")

        with gr.Row():
            system_dropdown = gr.Dropdown(
                choices=game_systems,
                value="(all)",
                label="Game System",
                scale=1,
            )

        additional_inputs = [system_dropdown]
        if stt_enabled:
            audio_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="Voice input (record, then send)",
            )
            additional_inputs.append(audio_input)

        gr.ChatInterface(
            fn=_chat_response,
            additional_inputs=additional_inputs,
        )

    demo.launch(
        server_port=settings.web.server_port,
        share=settings.web.share,
        allowed_paths=allowed_paths,
    )
