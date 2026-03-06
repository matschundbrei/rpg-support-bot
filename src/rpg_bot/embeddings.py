from __future__ import annotations

from openai import OpenAI

from rpg_bot.config import get_settings

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        settings = get_settings()
        _client = OpenAI(
            base_url=settings.embeddings.base_url,
            api_key="no-key-required",
            timeout=120.0,
        )
    return _client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts via any OpenAI-compatible embeddings API."""
    settings = get_settings()
    client = _get_client()

    response = client.embeddings.create(
        model=settings.embeddings.model,
        input=texts,
    )
    sorted_data = sorted(response.data, key=lambda x: x.index)
    return [item.embedding for item in sorted_data]


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([text])[0]
