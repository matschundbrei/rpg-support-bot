from __future__ import annotations

from collections.abc import Callable

from rpg_bot.config import get_settings
from rpg_bot.embeddings import embed_query
from rpg_bot.retrieval.store import VectorStore


def query_rag(
    user_query: str,
    game_system: str | None = None,
) -> str | None:
    """Embed query, retrieve relevant chunks, format as context string."""
    settings = get_settings()
    store = VectorStore()

    if store.count() == 0:
        return None

    query_embedding = embed_query(user_query)

    where = None
    if game_system:
        where = {"game_system": game_system}

    results = store.query(
        query_embedding=query_embedding,
        n_results=settings.retrieval.top_k,
        where=where,
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not documents:
        return None

    # Filter by relevance threshold (cosine distance: lower = more similar)
    threshold = settings.retrieval.relevance_threshold
    context_blocks: list[str] = []
    seen_texts: set[str] = set()

    for doc, meta, dist in zip(documents, metadatas, distances):
        if dist > threshold:
            continue

        # Deduplicate overlapping chunks (simple text prefix check)
        prefix = doc[:100]
        if prefix in seen_texts:
            continue
        seen_texts.add(prefix)

        source = meta.get("source", "Unknown")
        page = meta.get("page", "?")
        idx = len(context_blocks) + 1

        context_blocks.append(
            f"[{idx}] Source: {source}, p.{page}\n{doc}"
        )

    if not context_blocks:
        return None

    return "\n\n---\n\n".join(context_blocks)


def create_query_fn() -> Callable[[str, str | None], str | None] | None:
    """Create a query function if there are ingested documents, else return None."""
    try:
        store = VectorStore()
        if store.count() == 0:
            return None
    except Exception:
        return None

    return query_rag
