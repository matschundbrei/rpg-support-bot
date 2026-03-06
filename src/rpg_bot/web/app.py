from __future__ import annotations

from collections.abc import Generator

import gradio as gr

from rpg_bot.config import get_settings
from rpg_bot.llm.client import LLMClient
from rpg_bot.retrieval.query import query_rag
from rpg_bot.retrieval.store import VectorStore


def _get_game_systems() -> list[str]:
    try:
        store = VectorStore()
        sources = store.list_sources()
        # Extract unique game systems from metadata
        results = store.collection.get(include=["metadatas"], limit=10000)
        systems = set()
        for meta in results.get("metadatas", []):
            if meta and meta.get("game_system"):
                systems.add(meta["game_system"])
        return ["(all)"] + sorted(systems)
    except Exception:
        return ["(all)"]


def _chat_response(
    message: str,
    history: list[dict[str, str]],
    game_system: str,
) -> Generator[str, None, None]:
    llm = LLMClient()

    # Restore conversation history
    for msg in history:
        llm.history.append({"role": msg["role"], "content": msg["content"]})

    # RAG retrieval
    system_filter = game_system if game_system != "(all)" else None
    context = None
    try:
        context = query_rag(message, game_system=system_filter)
    except Exception:
        pass

    # Stream response
    collected = ""
    for chunk in llm.chat_stream(message, context=context):
        collected += chunk
        yield collected


def launch_app() -> None:
    settings = get_settings()
    game_systems = _get_game_systems()

    with gr.Blocks(title="RPG Support Bot") as demo:
        gr.Markdown("# RPG Support Bot\nAsk questions about RPG rules, world building, characters, and more.")

        with gr.Row():
            system_dropdown = gr.Dropdown(
                choices=game_systems,
                value="(all)",
                label="Game System",
                scale=1,
            )

        gr.ChatInterface(
            fn=_chat_response,
            type="messages",
            additional_inputs=[system_dropdown],
        )

    demo.launch(
        server_port=settings.web.server_port,
        share=settings.web.share,
    )
