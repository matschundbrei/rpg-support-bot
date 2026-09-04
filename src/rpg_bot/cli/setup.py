"""Interactive setup wizard: writes config.yaml and .env with user-chosen values."""

from __future__ import annotations

import getpass
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.panel import Panel

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

console = Console()

_BACKENDS = ["anthropic", "openai"]
_MODEL_DEFAULTS = {"anthropic": "claude-sonnet-4-6", "openai": "gpt-4o-mini"}


def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    return input(f"{prompt}{suffix}: ").strip() or default


def _ask_choice(prompt: str, choices: list[str], default: str) -> str:
    while True:
        value = _ask(f"{prompt} ({' / '.join(choices)})", default)
        if value in choices:
            return value
        console.print(f"[red]Please pick one of: {' / '.join(choices)}[/red]")


def _ask_int(prompt: str, default: int, *, lo: int = 0, hi: int | None = None) -> int:
    while True:
        try:
            value = int(_ask(prompt, str(default)))
        except ValueError:
            console.print("[red]Please enter a whole number.[/red]")
            continue
        if lo <= value and (hi is None or value <= hi):
            return value
        console.print(f"[red]Must be between {lo} and {hi}.[/red]")


def _ask_float(prompt: str, default: float, *, lo: float = 0.0, hi: float = 2.0) -> float:
    while True:
        try:
            value = float(_ask(prompt, str(default)))
        except ValueError:
            console.print("[red]Please enter a number.[/red]")
            continue
        if lo <= value <= hi:
            return value
        console.print(f"[red]Must be between {lo} and {hi}.[/red]")


def _ask_secret(name: str, existing: str) -> str:
    hint = f"current: {mask_key(existing)}" if existing else "not set"
    blank_action = "keep current" if existing else "leave empty"
    prompt = f"{name} ({hint}) — enter new value or blank to {blank_action}: "
    return getpass.getpass(prompt).strip()


def mask_key(value: str) -> str:
    """Mask a key for display, e.g. 'sk-ant-...abc123'."""
    if len(value) >= 12:
        return f"{value[:4]}...{value[-4:]}"
    if value:
        return "(set)"
    return "(empty)"


def _load_existing_config() -> dict[str, Any]:
    path = _PROJECT_ROOT / "config.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_existing_env() -> dict[str, str]:
    env: dict[str, str] = {}
    path = _PROJECT_ROOT / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip()
    return env


def _ask_llm(current: dict[str, Any]) -> dict[str, Any]:
    console.print("[bold cyan]LLM[/bold cyan]")
    backend = _ask_choice("Backend", _BACKENDS, str(current.get("backend", "anthropic")))
    model_default = str(current.get("model") or _MODEL_DEFAULTS[backend])
    cfg: dict[str, Any] = {
        "backend": backend,
        "model": _ask("Model", model_default),
        "base_url": "",
        "max_tokens": _ask_int("Max tokens", int(current.get("max_tokens", 4096))),
        "temperature": _ask_float("Temperature", float(current.get("temperature", 0.3))),
        "max_history": _ask_int("Max history messages", int(current.get("max_history", 20))),
    }
    if backend == "openai":
        cfg["base_url"] = _ask(
            "OpenAI-compatible base URL (blank = real OpenAI API, "
            "e.g. http://localhost:1234/v1 for LM Studio, http://localhost:11434/v1 for Ollama)",
            str(current.get("base_url", "")),
        )
    return cfg


def _ask_embeddings(current: dict[str, Any]) -> dict[str, Any]:
    console.print("[bold cyan]Embeddings[/bold cyan]")
    return {
        "model": _ask("Model", str(current.get("model", "nomic-embed-text-v2-moe:latest"))),
        "base_url": _ask(
            "OpenAI-compatible /v1 URL (blank = real OpenAI API, e.g. http://localhost:11434/v1)",
            str(current.get("base_url", "http://localhost:11434/v1")),
        ),
    }


def _ask_chunking(current: dict[str, Any]) -> dict[str, Any]:
    console.print("[bold cyan]Chunking[/bold cyan]")
    size = _ask_int("Chunk size (chars)", int(current.get("chunk_size", 1500)), lo=100)
    while True:
        overlap = _ask_int("Chunk overlap (chars)", int(current.get("chunk_overlap", 200)))
        if overlap < size:
            break
        console.print(f"[red]Overlap must be smaller than the chunk size ({size}).[/red]")
    return {"chunk_size": size, "chunk_overlap": overlap}


def _ask_retrieval(current: dict[str, Any]) -> dict[str, Any]:
    console.print("[bold cyan]Retrieval[/bold cyan]")
    return {
        "top_k": _ask_int("Top-k results", int(current.get("top_k", 15))),
        "relevance_threshold": _ask_float(
            "Relevance threshold (cosine distance, higher = stricter)",
            float(current.get("relevance_threshold", 1.0)),
            lo=0.0,
            hi=4.0,
        ),
    }


def _ask_storage(current: dict[str, Any], sourcebooks_default: str) -> tuple[dict[str, Any], str]:
    console.print("[bold cyan]Storage & source books[/bold cyan]")
    chromadb = {
        "persist_directory": _ask(
            "ChromaDB persist directory", str(current.get("persist_directory", "data/chromadb"))
        ),
        "collection_name": _ask(
            "ChromaDB collection name", str(current.get("collection_name", "rpg_sourcebooks"))
        ),
    }
    sourcebooks = _ask(
        "Source books directory (PDFs, one subdirectory per game system)",
        sourcebooks_default or "sourcebooks",
    )
    return chromadb, sourcebooks


def _ask_secrets(existing: dict[str, str]) -> dict[str, str]:
    console.print("[bold cyan]API keys (stored in .env)[/bold cyan]")
    return {
        "ANTHROPIC_API_KEY": _ask_secret(
            "Anthropic API key", existing.get("ANTHROPIC_API_KEY", "")
        ),
        "OPENAI_API_KEY": _ask_secret("OpenAI API key", existing.get("OPENAI_API_KEY", "")),
        "API_KEY": _ask_secret(
            "API server auth key (optional, protects /v1 and /api)",
            existing.get("API_KEY", ""),
        ),
    }


def _q(value: str) -> str:
    """Double-quote a string for YAML output."""
    return '"' + value.replace('"', '\\"') + '"'


def build_config_yaml(cfg: dict[str, Any]) -> str:
    """Render the wizard result as config.yaml text (config.example.yaml shape)."""
    llm = cfg["llm"]
    emb = cfg["embeddings"]
    chunk = cfg["chunking"]
    retrieval = cfg["retrieval"]
    chroma = cfg["chromadb"]
    return "\n".join(
        [
            "llm:",
            f"  backend: {_q(llm['backend'])}",
            f"  model: {_q(llm['model'])}",
            f"  base_url: {_q(llm.get('base_url', ''))}",
            f"  max_tokens: {llm['max_tokens']}",
            f"  temperature: {llm['temperature']}",
            f"  max_history: {llm['max_history']}",
            "",
            "embeddings:",
            f"  model: {_q(emb['model'])}",
            f"  base_url: {_q(emb['base_url'])}",
            "",
            "chunking:",
            f"  chunk_size: {chunk['chunk_size']}",
            f"  chunk_overlap: {chunk['chunk_overlap']}",
            "",
            "retrieval:",
            f"  top_k: {retrieval['top_k']}",
            f"  relevance_threshold: {retrieval['relevance_threshold']}",
            "",
            "chromadb:",
            f"  persist_directory: {_q(chroma['persist_directory'])}",
            f"  collection_name: {_q(chroma['collection_name'])}",
            "",
            f"sourcebooks_directory: {_q(cfg['sourcebooks_directory'])}",
            "",
        ]
    )


def build_env_text(env: dict[str, str]) -> str:
    """Render the wizard result as .env text (.env.example shape)."""
    return "\n".join(
        [
            f"ANTHROPIC_API_KEY={env.get('ANTHROPIC_API_KEY', '')}",
            f"OPENAI_API_KEY={env.get('OPENAI_API_KEY', '')}",
            "",
            "# Optional: protect the API server (/v1 and /api). Leave empty for no auth.",
            "# When set, clients must send: Authorization: Bearer <API_KEY>",
            f"API_KEY={env.get('API_KEY', '')}",
            "",
        ]
    )


def _print_summary(cfg: dict[str, Any], env: dict[str, str]) -> None:
    console.print(Panel(build_config_yaml(cfg), title="config.yaml", border_style="green"))
    env_preview = "\n".join(
        f"{key}={mask_key(env.get(key, ''))}"
        for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "API_KEY")
    )
    console.print(Panel(env_preview, title=".env (masked)", border_style="green"))


def run_setup() -> None:
    console.print(
        Panel(
            "Interactive setup for the RPG support bot.\n\n"
            "This writes [bold]config.yaml[/bold] (LLM, embeddings, chunking, retrieval) "
            "and [bold].env[/bold] (API keys).\n"
            "Press Enter to accept the [default] value in brackets.",
            title="RPG Support Bot — Setup",
            border_style="blue",
        )
    )

    try:
        existing_cfg = _load_existing_config()
        existing_env = _load_existing_env()
        if existing_cfg or existing_env:
            console.print(
                "[dim]Existing configuration found — current values are pre-filled; "
                "saving will overwrite the files.[/dim]"
            )

        cfg: dict[str, Any] = {
            "llm": _ask_llm(existing_cfg.get("llm", {})),
            "embeddings": _ask_embeddings(existing_cfg.get("embeddings", {})),
            "chunking": _ask_chunking(existing_cfg.get("chunking", {})),
            "retrieval": _ask_retrieval(existing_cfg.get("retrieval", {})),
        }
        cfg["chromadb"], cfg["sourcebooks_directory"] = _ask_storage(
            existing_cfg.get("chromadb", {}), str(existing_cfg.get("sourcebooks_directory", ""))
        )
        env = _ask_secrets(existing_env)

        _print_summary(cfg, env)
        answer = input("\nWrite these files? [y/N]: ").strip().lower()
        if answer not in ("y", "yes"):
            console.print("[yellow]Aborted — no files were written.[/yellow]")
            return

        (_PROJECT_ROOT / "config.yaml").write_text(build_config_yaml(cfg))
        (_PROJECT_ROOT / ".env").write_text(build_env_text(env))

        console.print("[green]Wrote config.yaml and .env.[/green]")
        console.print(
            "Next steps:\n"
            "  1. uv run rpg-bot ingest   # ingest PDFs from the source books directory\n"
            "  2. uv run rpg-bot chat     # start chatting"
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Aborted — no files were written.[/yellow]")
