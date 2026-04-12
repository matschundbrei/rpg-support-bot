from __future__ import annotations

from openai import OpenAI

from rpg_bot.config import get_settings

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        settings = get_settings()
        kwargs = {}
        if settings.embeddings.base_url:
            kwargs["base_url"] = settings.embeddings.base_url
        _client = OpenAI(
            api_key=settings.openai_api_key or "no-key-required",
            timeout=600.0,
            **kwargs,
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
