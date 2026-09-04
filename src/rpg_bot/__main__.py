from __future__ import annotations

import argparse


def cmd_chat(args: argparse.Namespace) -> None:
    from rpg_bot.cli.chat import run_chat
    from rpg_bot.retrieval.query import create_query_fn

    query_fn = create_query_fn()
    run_chat(query_fn=query_fn, game_system=args.system)


def cmd_ingest(args: argparse.Namespace) -> None:
    from rpg_bot.ingest.embed import ingest_sourcebooks

    ingest_sourcebooks(path=args.path)


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from rpg_bot.api.server import app, configure_cors

    if args.cors_origins:
        origins = [o.strip() for o in args.cors_origins.split(",")]
        configure_cors(origins)

    host = args.host
    display_host = "localhost" if host in ("0.0.0.0", "127.0.0.1") else host
    print(f"RPG Support Bot running at http://{display_host}:{args.port}")
    print(f"OpenAI-compatible API at http://{display_host}:{args.port}/v1")
    uvicorn.run(app, host=host, port=args.port)


def cmd_list(args: argparse.Namespace) -> None:
    from rpg_bot.retrieval.store import get_store

    store = get_store()
    sources = store.list_sources()
    if not sources:
        print("No source books ingested yet. Run: rpg-bot ingest")
        return
    print(f"Ingested source books ({len(sources)}):")
    for src in sources:
        print(f"  - {src}")


def cmd_setup(args: argparse.Namespace) -> None:
    from rpg_bot.cli.setup import run_setup

    run_setup()


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

    serve_p = sub.add_parser("serve", help="Launch web UI and OpenAI-compatible API server")
    serve_p.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    serve_p.add_argument("--port", "-p", type=int, default=8000, help="Port (default: 8000)")
    serve_p.add_argument(
        "--cors-origins",
        help="Comma-separated list of allowed CORS origins (e.g. 'http://localhost:3000,http://localhost:8080')",
    )

    sub.add_parser("list", help="List ingested source books")
    sub.add_parser("setup", help="Interactive configuration wizard (writes config.yaml and .env)")

    args = parser.parse_args()

    if args.command is None:
        # Default to chat
        args.system = None
        cmd_chat(args)
    elif args.command == "chat":
        cmd_chat(args)
    elif args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "setup":
        cmd_setup(args)


if __name__ == "__main__":
    main()
