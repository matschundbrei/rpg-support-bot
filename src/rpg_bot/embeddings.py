from __future__ import annotations

import httpx

from rpg_bot.config import get_settings

_client: httpx.Client | None = None


def _get_client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(timeout=120.0)
    return _client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts via the LM Studio OpenAI-compatible API."""
    settings = get_settings()
    client = _get_client()

    response = client.post(
        f"{settings.embeddings.base_url}/v1/embeddings",
        json={
            "model": settings.embeddings.model,
            "input": texts,
        },
    )
    response.raise_for_status()
    data = response.json()
    # Sort by index to ensure order matches input
    sorted_data = sorted(data["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in sorted_data]


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([text])[0]
