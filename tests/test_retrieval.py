import types

import pytest

import rpg_bot.retrieval.query as query_module
from rpg_bot.retrieval.query import (
    _reciprocal_rank_fusion,
    create_query_fn,
    query_rag,
)
from rpg_bot.retrieval.store import VectorStore, reset_store


@pytest.fixture(autouse=True)
def _reset_caches(monkeypatch):
    """Reset module-level BM25 cache and store singleton between tests."""
    reset_store()
    monkeypatch.setattr(query_module, "_bm25_index", None)
    monkeypatch.setattr(query_module, "_bm25_doc_ids", None)
    monkeypatch.setattr(query_module, "_bm25_metadatas", None)
    monkeypatch.setattr(query_module, "_bm25_corpus_size", None)
    monkeypatch.setattr(query_module, "_bm25_store_id", None)


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


def test_rrf_prefers_documents_ranked_high_in_both():
    vector = {"a": 0, "b": 1, "c": 2}
    keyword = {"a": 1, "c": 0, "b": 2}
    fused = _reciprocal_rank_fusion(vector, keyword)
    assert fused[0] == "a"
    assert set(fused) == {"a", "b", "c"}


def test_query_rag_hybrid(store, monkeypatch):
    store.add(
        ids=["q1"],
        documents=["[PHB > Initiative]\nInitiative is Reaction + Intuition."],
        embeddings=[[0.1] * 384],
        metadatas=[
            {
                "source": "PHB",
                "page": 71,
                "source_path": "/tmp/phb.pdf",
                "game_system": "dnd5e",
                "language": "en",
                "section": "Initiative",
                "breadcrumb": "PHB > Initiative",
            }
        ],
    )
    store.add(
        ids=["q2"],
        documents=["[DMG > Treasure]\nTreasure tables for dungeons."],
        embeddings=[[0.2] * 384],
        metadatas=[
            {
                "source": "DMG",
                "page": 10,
                "source_path": "/tmp/dmg.pdf",
                "game_system": "dnd5e",
                "language": "en",
                "section": "Treasure",
                "breadcrumb": "DMG > Treasure",
            }
        ],
    )

    fake_settings = types.SimpleNamespace(
        retrieval=types.SimpleNamespace(top_k=2, relevance_threshold=1.0),
    )
    monkeypatch.setattr(query_module, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(query_module, "embed_query", lambda q: [0.1] * 384)
    monkeypatch.setattr(query_module, "get_store", lambda: store)

    res = query_rag("Initiative", game_system="dnd5e")
    assert res is not None
    assert "PHB" in res.context
    assert "PHB, p.71" in res.source_map
    assert res.source_map["PHB, p.71"].startswith("file:///tmp/phb.pdf#page=71")


def test_query_rag_filters_by_game_system(store, monkeypatch):
    store.add(
        ids=["s1"],
        documents=["[SR6 > Initiative]\nInitiative is Reaction + Intuition."],
        embeddings=[[0.1] * 384],
        metadatas=[
            {
                "source": "SR6",
                "page": 71,
                "source_path": "/tmp/sr6.pdf",
                "game_system": "shadowrun6",
                "language": "de",
                "section": "Initiative",
                "breadcrumb": "SR6 > Initiative",
            }
        ],
    )

    fake_settings = types.SimpleNamespace(
        retrieval=types.SimpleNamespace(top_k=2, relevance_threshold=1.0),
    )
    monkeypatch.setattr(query_module, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(query_module, "embed_query", lambda q: [0.1] * 384)
    monkeypatch.setattr(query_module, "get_store", lambda: store)

    # No matching chunks for a different system
    assert query_rag("Initiative", game_system="dnd5e") is None


def test_create_query_fn_none_on_empty_store(tmp_path, monkeypatch):
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
                        "persist_path": tmp_path / "chromadb2",
                        "collection_name": "test_empty",
                    },
                )()
            },
        )(),
    )
    assert create_query_fn() is None
