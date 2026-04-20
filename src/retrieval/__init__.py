"""
src/retrieval — Long COVID RAG retrieval package.

Modules:
    models.py         — Shared Pydantic types
    config.py         — RetrievalConfig and sub-configs
    query_rewriter.py — Interactive LLM-based query rewrite + intent classification
    hybrid_search.py  — BYOV query embedding + Weaviate hybrid search (Phase 3b)
    reranker.py       — Toggleable cross-encoder reranker (Phase 3c)
    ranking.py        — Signal-weighted ranking with diversity cap (Phase 3d)
    pipeline.py       — Wires all modules into a single retrieve() call (Phase 3e)
"""
from .config import RetrievalConfig
from .hybrid_search import dedup_results, search
from .ranking import rank
from .reranker import rerank
from .models import (
    ChunkMetadata,
    IntentCategory,
    RankingPreset,
    RetrievalResult,
    RewriteMode,
    RewriteResult,
    SearchResult,
)

__all__ = [
    "RetrievalConfig",
    "search",
    "dedup_results",
    "rerank",
    "rank",
    "ChunkMetadata",
    "IntentCategory",
    "RankingPreset",
    "RetrievalResult",
    "RewriteMode",
    "RewriteResult",
    "SearchResult",
]
