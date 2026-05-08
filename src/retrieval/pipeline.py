"""
pipeline.py — Single-call retrieval: rewrite → hybrid search → rerank → rank.

Wires Phase 3a–3d into a ``RetrievalResult`` for synthesis (Phase 4) and the
future FastAPI layer (Phase 6). Uses ``best_rewrite.query`` for search and
rerank so ``RetrievalResult.query`` carries both ``original_query`` and the
rewrite text expected by synthesis prompts.

``search()`` accepts an optional Weaviate client; pass a long-lived client from
the API process to avoid connect/disconnect per request.
"""
from __future__ import annotations

import logging
import time

import weaviate

from .config import RetrievalConfig
from .hybrid_search import search as hybrid_search
from .models import RankingPreset, RetrievalResult, RewriteResult
from .query_rewriter import rewrite
from .ranking import rank
from .reranker import rerank

log = logging.getLogger("retrieval.pipeline")


def retrieve_from_rewrite(
    rewrite_result: RewriteResult,
    *,
    search_query: str | None = None,
    cfg: RetrievalConfig | None = None,
    preset: RankingPreset = RankingPreset.MOST_RELEVANT,
    voyage_api_key: str | None = None,
    weaviate_client: weaviate.WeaviateClient | None = None,
) -> RetrievalResult:
    """
    Run hybrid search → rerank → rank using a finalized ``RewriteResult``.

    Use when the rewriter has already run (or after the user resolves
    clarification) so retrieval uses an explicit search string while
    ``rewrite_result`` is still passed to synthesis prompts.

    Parameters
    ----------
    rewrite_result:
        Rewrite output to attach to ``RetrievalResult.query``.
    search_query:
        Text for embedding + BM25 + reranker. Defaults to
        ``rewrite_result.best_rewrite.query``.
    cfg:
        ``RetrievalConfig`` for search / rerank / rank.
    preset:
        Carried on ``RetrievalResult`` for analytics.
    voyage_api_key:
        Passed to hybrid search; falls back to ``VOYAGE_API_KEY``.
    weaviate_client:
        Optional long-lived Weaviate client (recommended for FastAPI).
    """
    cfg = cfg or RetrievalConfig()
    qtext = (search_query if search_query is not None else rewrite_result.best_rewrite.query).strip()
    t0 = time.perf_counter()

    hybrid_results = hybrid_search(
        qtext,
        cfg=cfg,
        voyage_api_key=voyage_api_key,
        weaviate_client=weaviate_client,
    )
    reranked = rerank(qtext, hybrid_results, cfg=cfg)
    ranked = rank(reranked, cfg=cfg.ranking)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    log.info(
        "retrieve_from_rewrite() done in %.0fms — %d chunks (reranker=%s)",
        elapsed_ms,
        len(ranked),
        cfg.reranker.enabled,
    )

    return RetrievalResult(
        query=rewrite_result,
        results=ranked,
        preset=preset,
        reranker_enabled=cfg.reranker.enabled,
        elapsed_ms=elapsed_ms,
    )


def retrieve(
    user_query: str,
    *,
    cfg: RetrievalConfig | None = None,
    preset: RankingPreset = RankingPreset.MOST_RELEVANT,
    voyage_api_key: str | None = None,
    openrouter_api_key: str | None = None,
    weaviate_client: weaviate.WeaviateClient | None = None,
) -> RetrievalResult:
    """
    Run the full retrieval stack for one user query.

    Parameters
    ----------
    user_query:
        Raw query string (stored on ``RewriteResult.original_query``).
    cfg:
        ``RetrievalConfig`` (search, reranker, ranking, rewriter). Defaults to
        a fresh ``RetrievalConfig()`` — enable ``cfg.reranker.enabled`` for
        cross-encoder reranking (recommended before synthesis).
    preset:
        UI / analytics preset carried on ``RetrievalResult``; does not change
        ranking weights (those come from ``cfg.ranking``).
    voyage_api_key:
        Passed to hybrid search; falls back to ``VOYAGE_API_KEY``.
    openrouter_api_key:
        Passed to the rewriter; falls back to ``OPENROUTER_API_KEY``.
    weaviate_client:
        Optional Weaviate client. When omitted, ``hybrid_search.search`` builds
        and closes its own client for this call.

    Returns
    -------
    RetrievalResult
        ``query`` is the ``RewriteResult``; ``results`` are ranked chunks.
    """
    cfg = cfg or RetrievalConfig()
    rewrite_result = rewrite(user_query, cfg=cfg, api_key=openrouter_api_key)
    return retrieve_from_rewrite(
        rewrite_result,
        search_query=rewrite_result.best_rewrite.query,
        cfg=cfg,
        preset=preset,
        voyage_api_key=voyage_api_key,
        weaviate_client=weaviate_client,
    )
