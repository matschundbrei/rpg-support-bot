import uuid

import pytest

from rpg_bot.retrieval.store import VectorStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Create a VectorStore with a temporary directory."""
    monkeypatch.setattr(
        "rpg_bot.retrieval.store.get_settings",
        lambda: type(
            "S",
            (),
            {
                "chromadb": type(
                    "C",
                    (),
                    {
                        "persist_path": tmp_path / "chromadb",
                        "collection_name": "test_collection",
                    },
                )()
            },
        )(),
    )
    return VectorStore()


def test_add_and_count(store):
    assert store.count() == 0
    store.add(
        ids=["id1", "id2"],
        documents=["doc one", "doc two"],
        embeddings=[[0.1] * 384, [0.2] * 384],
        metadatas=[{"source": "test"}, {"source": "test"}],
    )
    assert store.count() == 2


def test_list_sources(store):
    store.add(
        ids=["a", "b"],
        documents=["x", "y"],
        embeddings=[[0.0] * 384, [0.0] * 384],
        metadatas=[{"source": "PHB"}, {"source": "DMG"}],
    )
    sources = store.list_sources()
    assert "PHB" in sources
    assert "DMG" in sources


def test_query(store):
    store.add(
        ids=["c1"],
        documents=["The fighter has a high armor class"],
        embeddings=[[0.5] * 384],
        metadatas=[{"source": "PHB", "page": 42}],
    )
    results = store.query(query_embedding=[0.5] * 384, n_results=1)
    assert results["documents"][0][0] == "The fighter has a high armor class"


def test_has_source(store):
    assert not store.has_source("abc123")
    store.add(
        ids=["x1"],
        documents=["test"],
        embeddings=[[0.0] * 384],
        metadatas=[{"source": "PHB", "source_hash": "abc123"}],
    )
    assert store.has_source("abc123")


def test_delete_source(store):
    store.add(
        ids=["d1", "d2"],
        documents=["a", "b"],
        embeddings=[[0.0] * 384, [0.0] * 384],
        metadatas=[{"source": "PHB"}, {"source": "DMG"}],
    )
    assert store.count() == 2
    store.delete_source("PHB")
    assert store.count() == 1
    assert store.list_sources() == ["DMG"]
