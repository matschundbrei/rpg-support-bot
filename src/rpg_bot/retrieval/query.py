from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from urllib.parse import quote

import numpy as np
from rank_bm25 import BM25Okapi

from rpg_bot.config import get_settings
from rpg_bot.embeddings import embed_query
from rpg_bot.retrieval.store import VectorStore


@dataclass
class RAGResult:
    context: str
    source_map: dict[str, str] = field(default_factory=dict)
    """Maps 'source_name, p.XX' -> file:// URL with page anchor."""


def _build_file_url(source_path: str, page: int | str) -> str:
    """Build a file:// URL that opens a PDF at a specific page."""
    if not source_path:
        return ""
    return f"file://{quote(source_path, safe='/')}#page={page}"


# --- BM25 keyword index (cached at module level) ---

_bm25_index: BM25Okapi | None = None
_bm25_doc_ids: list[str] | None = None
_bm25_metadatas: list[dict] | None = None
_bm25_corpus_size: int | None = None

_TOKENIZE_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [w for w in _TOKENIZE_RE.findall(text.lower()) if len(w) > 2]


def _get_bm25_index(store: VectorStore) -> tuple[BM25Okapi, list[str], list[dict]]:
    global _bm25_index, _bm25_doc_ids, _bm25_metadatas, _bm25_corpus_size

    current_size = store.count()
    if _bm25_index is not None and _bm25_corpus_size == current_size:
        return _bm25_index, _bm25_doc_ids, _bm25_metadatas

    all_data = store.collection.get(include=["documents", "metadatas"])
    _bm25_doc_ids = all_data["ids"]
    _bm25_metadatas = all_data["metadatas"]

    tokenized = [_tokenize(doc) for doc in all_data["documents"]]
    _bm25_index = BM25Okapi(tokenized)
    _bm25_corpus_size = current_size

    return _bm25_index, _bm25_doc_ids, _bm25_metadatas


def _reciprocal_rank_fusion(
    vector_ranking: dict[str, int],
    keyword_ranking: dict[str, int],
    k: int = 60,
) -> list[str]:
    """Combine two rankings using RRF. Returns IDs sorted by fused score."""
    scores: dict[str, float] = {}
    for doc_id in set(vector_ranking) | set(keyword_ranking):
        score = 0.0
        if doc_id in vector_ranking:
            score += 1.0 / (k + vector_ranking[doc_id])
        if doc_id in keyword_ranking:
            score += 1.0 / (k + keyword_ranking[doc_id])
        scores[doc_id] = score
    return sorted(scores, key=lambda x: scores[x], reverse=True)


def query_rag(
    user_query: str,
    game_system: str | None = None,
) -> RAGResult | None:
    """Hybrid search: vector similarity + BM25 keyword matching with RRF fusion."""
    settings = get_settings()
    store = VectorStore()

    if store.count() == 0:
        return None

    top_k = settings.retrieval.top_k
    n_candidates = top_k * 3

    where = None
    if game_system:
        where = {"game_system": game_system}

    # 1. Vector search
    query_embedding = embed_query(user_query)
    vector_results = store.query(
        query_embedding=query_embedding,
        n_results=n_candidates,
        where=where,
    )

    vector_ids = vector_results.get("ids", [[]])[0]
    vector_distances = vector_results.get("distances", [[]])[0]
    vector_ranking = {doc_id: rank for rank, doc_id in enumerate(vector_ids)}
    id_to_dist = dict(zip(vector_ids, vector_distances))

    # 2. BM25 keyword search
    bm25, bm25_ids, bm25_metas = _get_bm25_index(store)
    query_tokens = _tokenize(user_query)

    if query_tokens:
        bm25_scores = bm25.get_scores(query_tokens)

        # Apply game_system filter
        if game_system:
            for i, meta in enumerate(bm25_metas):
                if meta.get("game_system") != game_system:
                    bm25_scores[i] = 0.0

        top_bm25_idx = np.argsort(bm25_scores)[::-1][:n_candidates]
        keyword_ranking = {}
        for rank, idx in enumerate(top_bm25_idx):
            if bm25_scores[idx] > 0:
                keyword_ranking[bm25_ids[idx]] = rank
    else:
        keyword_ranking = {}

    # 3. RRF fusion
    fused_ids = _reciprocal_rank_fusion(vector_ranking, keyword_ranking)[:top_k]

    if not fused_ids:
        return None

    # 4. Fetch documents for fused results
    fetched = store.collection.get(ids=fused_ids, include=["documents", "metadatas"])
    id_to_doc = dict(zip(fetched["ids"], fetched["documents"]))
    id_to_meta = dict(zip(fetched["ids"], fetched["metadatas"]))

    # 5. Build context, dedup, apply threshold only to vector-only results
    threshold = settings.retrieval.relevance_threshold
    context_blocks: list[str] = []
    source_map: dict[str, str] = {}
    seen_texts: set[str] = set()

    for doc_id in fused_ids:
        doc = id_to_doc.get(doc_id)
        meta = id_to_meta.get(doc_id)
        if not doc or not meta:
            continue

        # Deduplicate
        prefix = doc[:100]
        if prefix in seen_texts:
            continue
        seen_texts.add(prefix)

        # For results found only by vector search, apply distance threshold
        is_keyword_match = doc_id in keyword_ranking
        dist = id_to_dist.get(doc_id)
        if not is_keyword_match and dist is not None and dist > threshold:
            continue

        source = meta.get("source", "Unknown")
        page = meta.get("page", "?")
        source_path = meta.get("source_path", "")
        idx = len(context_blocks) + 1

        context_blocks.append(
            f"[{idx}] Source: {source}, p.{page}\n{doc}"
        )

        citation_key = f"{source}, p.{page}"
        if citation_key not in source_map and source_path:
            source_map[citation_key] = _build_file_url(source_path, page)

    if not context_blocks:
        return None

    return RAGResult(
        context="\n\n---\n\n".join(context_blocks),
        source_map=source_map,
    )


def create_query_fn() -> Callable[[str, str | None], RAGResult | None] | None:
    """Create a query function if there are ingested documents, else return None."""
    try:
        store = VectorStore()
        if store.count() == 0:
            return None
    except Exception:
        return None

    return query_rag
