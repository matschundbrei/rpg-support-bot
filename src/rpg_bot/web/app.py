from __future__ import annotations

import re
from collections.abc import Generator

import gradio as gr

from rpg_bot.config import get_settings
from rpg_bot.llm.client import LLMClient
from rpg_bot.retrieval.query import query_rag
from rpg_bot.retrieval.store import VectorStore


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


def _chat_response(
    message: dict,
    history: list[dict[str, str]],
    game_system: str,
) -> Generator[str, None, None]:
    user_text = message.get("text", "") if isinstance(message, dict) else str(message)

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


_SPEECH_JS = """
function() {
    if (document.getElementById('speech-btn')) return;

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) return;

    const btn = document.createElement('button');
    btn.id = 'speech-btn';
    btn.textContent = '🎤';
    btn.title = 'Speech to text (click to start, click again to stop)';
    btn.style.cssText = 'position:fixed;bottom:24px;right:24px;z-index:9999;' +
        'width:48px;height:48px;border-radius:50%;border:none;' +
        'font-size:24px;cursor:pointer;background:#f0f0f0;box-shadow:0 2px 8px rgba(0,0,0,0.2);';

    let recognition = null;
    let listening = false;

    btn.onclick = () => {
        if (listening) {
            recognition.stop();
            return;
        }
        recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = document.documentElement.lang || navigator.language || 'en';

        const textarea = document.querySelector('textarea');
        if (!textarea) return;
        const startValue = textarea.value;

        recognition.onresult = (e) => {
            let transcript = '';
            for (let i = 0; i < e.results.length; i++) {
                transcript += e.results[i][0].transcript;
            }
            const nativeSet = Object.getOwnPropertyDescriptor(
                window.HTMLTextAreaElement.prototype, 'value').set;
            nativeSet.call(textarea, startValue + transcript);
            textarea.dispatchEvent(new Event('input', {bubbles: true}));
        };
        recognition.onstart = () => {
            listening = true;
            btn.style.background = '#ff4444';
            btn.style.color = 'white';
        };
        recognition.onend = () => {
            listening = false;
            btn.style.background = '#f0f0f0';
            btn.style.color = 'black';
        };
        recognition.onerror = () => {
            listening = false;
            btn.style.background = '#f0f0f0';
            btn.style.color = 'black';
        };
        recognition.start();
    };
    document.body.appendChild(btn);
}
"""


def launch_app() -> None:
    settings = get_settings()
    game_systems = _get_game_systems()
    allowed_paths = _get_source_paths()

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
            additional_inputs=[system_dropdown],
        )

        demo.load(None, js=_SPEECH_JS)

    demo.launch(
        server_port=settings.web.server_port,
        share=settings.web.share,
        allowed_paths=allowed_paths,
    )
