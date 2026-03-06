from __future__ import annotations

import chromadb
from chromadb.config import Settings as ChromaSettings

from rpg_bot.config import get_settings


class VectorStore:
    def __init__(self) -> None:
        settings = get_settings()
        persist_path = settings.chromadb.persist_path
        persist_path.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(persist_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection_name = settings.chromadb.collection_name

    @property
    def collection(self) -> chromadb.Collection:
        return self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 8,
        where: dict | None = None,
    ) -> dict:
        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        return self.collection.query(**kwargs)

    def list_sources(self) -> list[str]:
        try:
            results = self.collection.get(include=["metadatas"], limit=10000)
        except Exception:
            return []
        sources = set()
        for meta in results.get("metadatas", []):
            if meta and "source" in meta:
                sources.add(meta["source"])
        return sorted(sources)

    def has_source(self, source_hash: str) -> bool:
        """Check if a source with the given hash has already been ingested."""
        try:
            results = self.collection.get(
                where={"source_hash": source_hash},
                limit=1,
                include=[],
            )
            return len(results.get("ids", [])) > 0
        except Exception:
            return False

    def delete_source(self, source_name: str) -> None:
        try:
            self.collection.delete(where={"source": source_name})
        except Exception:
            pass

    def count(self) -> int:
        return self.collection.count()
