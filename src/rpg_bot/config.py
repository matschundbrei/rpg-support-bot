from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_yaml_config() -> dict[str, Any]:
    config_path = _PROJECT_ROOT / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


_yaml = _load_yaml_config()


class LLMConfig(BaseSettings):
    backend: str = _yaml.get("llm", {}).get("backend", "anthropic")
    model: str = _yaml.get("llm", {}).get("model", "claude-sonnet-4-20250514")
    max_tokens: int = _yaml.get("llm", {}).get("max_tokens", 4096)
    temperature: float = _yaml.get("llm", {}).get("temperature", 0.3)
    base_url: str = _yaml.get("llm", {}).get("base_url", "")
    # Max conversation messages kept in the rolling CLI history
    max_history: int = _yaml.get("llm", {}).get("max_history", 20)


class EmbeddingsConfig(BaseSettings):
    model: str = _yaml.get("embeddings", {}).get(
        "model", "nomic-embed-text-v2-moe:latest"
    )
    base_url: str = _yaml.get("embeddings", {}).get(
        "base_url", "http://localhost:11434/v1"
    )


class ChunkingConfig(BaseSettings):
    chunk_size: int = _yaml.get("chunking", {}).get("chunk_size", 1500)
    chunk_overlap: int = _yaml.get("chunking", {}).get("chunk_overlap", 200)


class RetrievalConfig(BaseSettings):
    top_k: int = _yaml.get("retrieval", {}).get("top_k", 15)
    relevance_threshold: float = _yaml.get("retrieval", {}).get(
        "relevance_threshold", 1.0
    )


class ChromaDBConfig(BaseSettings):
    persist_directory: str = _yaml.get("chromadb", {}).get(
        "persist_directory", "data/chromadb"
    )
    collection_name: str = _yaml.get("chromadb", {}).get(
        "collection_name", "rpg_sourcebooks"
    )

    @property
    def persist_path(self) -> Path:
        path = Path(self.persist_directory)
        if not path.is_absolute():
            path = _PROJECT_ROOT / path
        return path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: str = Field(default="")
    openai_api_key: str = Field(default="")
    # Optional bearer key protecting the API server (/v1 and /api endpoints).
    # Empty = no auth (default, backward compatible). Set via API_KEY in .env.
    api_key: str = Field(default="")

    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    chromadb: ChromaDBConfig = Field(default_factory=ChromaDBConfig)

    sourcebooks_directory: str = _yaml.get("sourcebooks_directory", "sourcebooks")

    @property
    def sourcebooks_path(self) -> Path:
        path = Path(self.sourcebooks_directory)
        if not path.is_absolute():
            path = _PROJECT_ROOT / path
        return path


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
