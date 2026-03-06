from __future__ import annotations

import re

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from rpg_bot.llm.client import LLMClient

console = Console()


def _print_welcome() -> None:
    console.print(
        Panel(
            "[bold]RPG Support Bot[/bold]\n"
            "Ask questions about RPG rules, world building, characters, and more.\n\n"
            "Commands: /clear, /sources, /system <name>, /quit",
            title="Welcome",
            border_style="blue",
        )
    )


_CITATION_RE = re.compile(r"\[([^]]+?,\s*p\.\s*\d+)\]")


def _render_citations(text: str, source_map: dict[str, str]) -> Text:
    """Convert citation patterns like [Book, p.42] into clickable terminal links."""
    rich_text = Text()
    last_end = 0

    for match in _CITATION_RE.finditer(text):
        # Add text before the citation
        rich_text.append(text[last_end:match.start()])

        citation_key = match.group(1)
        url = source_map.get(citation_key, "")

        if url:
            rich_text.append(f"[{citation_key}]", style=f"link {url} bold cyan")
        else:
            rich_text.append(f"[{citation_key}]", style="bold cyan")

        last_end = match.end()

    rich_text.append(text[last_end:])
    return rich_text


def run_chat(
    query_fn: callable | None = None,
    game_system: str | None = None,
) -> None:
    """Run the interactive CLI chat loop.

    Args:
        query_fn: Optional RAG query function (user_query, game_system) -> RAGResult.
        game_system: Optional game system filter for RAG queries.
    """
    _print_welcome()

    llm = LLMClient()
    session: PromptSession[str] = PromptSession(history=InMemoryHistory())
    current_system = game_system

    while True:
        try:
            user_input = session.prompt("\n[You] > ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "/quit":
            console.print("Goodbye!")
            break

        if user_input.lower() == "/clear":
            llm.clear_history()
            console.print("[dim]Conversation cleared.[/dim]")
            continue

        if user_input.lower() == "/sources":
            if query_fn is None:
                console.print("[dim]No source books ingested. Run: rpg-bot ingest[/dim]")
            else:
                console.print(f"[dim]Active game system filter: {current_system or 'none (all systems)'}[/dim]")
            continue

        if user_input.lower().startswith("/system"):
            parts = user_input.split(maxsplit=1)
            if len(parts) > 1:
                current_system = parts[1].strip()
                console.print(f"[dim]Game system set to: {current_system}[/dim]")
            else:
                current_system = None
                console.print("[dim]Game system filter cleared.[/dim]")
            continue

        # RAG context retrieval
        context = None
        source_map: dict[str, str] = {}
        if query_fn is not None:
            try:
                rag_result = query_fn(user_input, current_system)
                if rag_result is not None:
                    context = rag_result.context
                    source_map = rag_result.source_map
            except Exception as e:
                console.print(f"[yellow]RAG retrieval failed: {e}[/yellow]")

        # Stream LLM response with live markdown preview
        console.print()
        collected = ""
        try:
            with Live(Markdown(""), console=console, refresh_per_second=10) as live:
                for chunk in llm.chat_stream(user_input, context=context):
                    collected += chunk
                    live.update(Markdown(collected))

            # Re-render final response with clickable citation links
            if source_map:
                console.print()
                console.print(_render_citations(collected, source_map))
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            # Remove the failed user message from history
            if llm.history and llm.history[-1]["role"] == "user":
                llm.history.pop()
