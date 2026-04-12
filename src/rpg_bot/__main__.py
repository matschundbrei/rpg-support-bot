from __future__ import annotations

import argparse
import sys


def cmd_chat(args: argparse.Namespace) -> None:
    from rpg_bot.cli.chat import run_chat
    from rpg_bot.retrieval.query import create_query_fn

    query_fn = create_query_fn()
    run_chat(query_fn=query_fn, game_system=args.system)


def cmd_ingest(args: argparse.Namespace) -> None:
    from rpg_bot.ingest.embed import ingest_sourcebooks

    ingest_sourcebooks(path=args.path)


def cmd_web(args: argparse.Namespace) -> None:
    from rpg_bot.web.app import launch_app

    launch_app()


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from rpg_bot.api.server import app

    print(f"Starting rpg-bot API server on {args.host}:{args.port}")
    print("Connect Open WebUI to: http://{host}:{port}/v1".format(host=args.host, port=args.port))
    uvicorn.run(app, host=args.host, port=args.port)


def cmd_list(args: argparse.Namespace) -> None:
    from rpg_bot.retrieval.store import VectorStore

    store = VectorStore()
    sources = store.list_sources()
    if not sources:
        print("No source books ingested yet. Run: rpg-bot ingest")
        return
    print(f"Ingested source books ({len(sources)}):")
    for src in sources:
        print(f"  - {src}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rpg-bot",
        description="RAG-powered RPG support bot",
    )
    sub = parser.add_subparsers(dest="command")

    chat_p = sub.add_parser("chat", help="Interactive chat (CLI)")
    chat_p.add_argument("--system", "-s", help="Filter by game system")

    ingest_p = sub.add_parser("ingest", help="Ingest PDF source books")
    ingest_p.add_argument("--path", "-p", help="Path to a specific PDF or directory")

    sub.add_parser("web", help="Launch Gradio web UI")

    serve_p = sub.add_parser("serve", help="Launch OpenAI-compatible API server")
    serve_p.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    serve_p.add_argument("--port", "-p", type=int, default=8000, help="Port (default: 8000)")

    sub.add_parser("list", help="List ingested source books")

    args = parser.parse_args()

    if args.command is None:
        # Default to chat
        args.system = None
        cmd_chat(args)
    elif args.command == "chat":
        cmd_chat(args)
    elif args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "web":
        cmd_web(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "list":
        cmd_list(args)


if __name__ == "__main__":
    main()
