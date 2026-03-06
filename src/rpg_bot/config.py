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
    model: str = _yaml.get("llm", {}).get("model", "claude-sonnet-4-20250514")
    max_tokens: int = _yaml.get("llm", {}).get("max_tokens", 4096)
    temperature: float = _yaml.get("llm", {}).get("temperature", 0.3)


class EmbeddingsConfig(BaseSettings):
    model: str = _yaml.get("embeddings", {}).get(
        "model", "text-embedding-nomic-embed-text-v1.5"
    )
    base_url: str = _yaml.get("embeddings", {}).get(
        "base_url", "http://localhost:1234"
    )


class ChunkingConfig(BaseSettings):
    chunk_size: int = _yaml.get("chunking", {}).get("chunk_size", 1000)
    chunk_overlap: int = _yaml.get("chunking", {}).get("chunk_overlap", 200)


class RetrievalConfig(BaseSettings):
    top_k: int = _yaml.get("retrieval", {}).get("top_k", 8)
    relevance_threshold: float = _yaml.get("retrieval", {}).get(
        "relevance_threshold", 0.3
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


class WebConfig(BaseSettings):
    server_port: int = _yaml.get("web", {}).get("server_port", 7860)
    share: bool = _yaml.get("web", {}).get("share", False)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: str = Field(default="")

    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    chromadb: ChromaDBConfig = Field(default_factory=ChromaDBConfig)
    web: WebConfig = Field(default_factory=WebConfig)

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
