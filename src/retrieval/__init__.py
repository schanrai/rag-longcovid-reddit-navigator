"""
src/retrieval — Long COVID RAG retrieval package.

Modules:
    models.py         — Shared Pydantic types
    config.py         — RetrievalConfig and sub-configs
    query_rewriter.py — Interactive LLM-based query rewrite + intent classification
    hybrid_search.py  — BYOV query embedding + Weaviate hybrid search (Phase 3b)
    reranker.py       — Toggleable cross-encoder reranker (Phase 3c)
    ranking.py        — Intent-driven freshness/popularity ranking (Phase 3d)
    pipeline.py       — Wires all modules into a single retrieve() call (Phase 3e)
"""
from .config import RetrievalConfig
from .hybrid_search import search
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
    "rerank",
    "ChunkMetadata",
    "IntentCategory",
    "RankingPreset",
    "RetrievalResult",
    "RewriteMode",
    "RewriteResult",
    "SearchResult",
]
