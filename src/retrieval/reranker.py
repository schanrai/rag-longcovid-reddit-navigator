"""
reranker.py — Toggleable cross-encoder reranker (Phase 3c).

Takes the top-k candidates from hybrid search and re-scores each (query, chunk)
pair using a cross-encoder. Unlike bi-encoders (which encode independently),
a cross-encoder processes query and document together, enabling richer attention-
based comparison at the cost of higher latency.

Design decisions:
- Toggleable: reranker_enabled=False bypasses this module entirely.
- Lazy model loading: model is loaded once on first call and cached for the
  lifetime of the process. Avoids cold-start penalty on every request.
- Hard latency gate: if reranking exceeds latency_budget_ms, the original
  hybrid scores are returned unchanged with a warning logged.
- Default model: cross-encoder/ms-marco-MiniLM-L-6-v2 (CPU-sufficient for
  top-50 reranking; ~50ms per batch on modern hardware).

CLI usage:
    python -m src.retrieval.reranker --query "fatigue long covid" --test
    python -m src.retrieval.reranker --query "..." --top-k 25 --verbose
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Final

from dotenv import load_dotenv

from .config import RetrievalConfig
from .hybrid_search import TEST_QUERIES, search as hybrid_search
from .models import SearchResult

load_dotenv()
log = logging.getLogger("retrieval.reranker")

DEFAULT_MODEL: Final[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Module-level model cache — loaded once, reused across calls
_model_cache: dict[str, object] = {}


def _load_model(model_name: str) -> object:
    """Load and cache a CrossEncoder model. Thread-safe for single-process use."""
    if model_name not in _model_cache:
        from sentence_transformers import CrossEncoder  # deferred import
        log.info("Loading cross-encoder model: %s", model_name)
        t0 = time.perf_counter()
        _model_cache[model_name] = CrossEncoder(model_name)
        elapsed = (time.perf_counter() - t0) * 1000
        log.info("Model loaded in %.0fms", elapsed)
    return _model_cache[model_name]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compose_rerank_text(r: SearchResult) -> str:
    """
    Build the document-side text passed to the cross-encoder.

    Uses post_title + chunk body only. post_summary is intentionally excluded:
    A/B testing (Finding 10, reranker_qa.md) showed that including post_summary
    inflated low-substance comments from dominant threads and suppressed diversity.
    Removing it improved top-10 thread diversity from 3→5 (Q7) and 4→9 (Q5).

    The embedding index still uses title + summary + body (compose_embedding_input
    in embed_eval.py) — this change is reranker-only.
    """
    m = r.metadata
    title = (m.post_title or "").strip()
    body = r.text.strip()

    parts = [title, body]

    return "\n\n".join(p for p in parts if p)


# ── Public entry point ────────────────────────────────────────────────────────

def rerank(
    query: str,
    results: list[SearchResult],
    *,
    cfg: RetrievalConfig | None = None,
) -> list[SearchResult]:
    """
    Re-score and reorder search results using a cross-encoder.

    If reranker is disabled in cfg, returns results unchanged.
    If reranking exceeds the latency budget, returns original results with a warning.

    Parameters
    ----------
    query:
        The rewritten query string (same string used for hybrid search).
    results:
        Candidates from hybrid_search.search() — typically top 50.
    cfg:
        RetrievalConfig; uses defaults if not provided.

    Returns
    -------
    list[SearchResult] sorted by rerank_score descending, truncated to
    cfg.reranker.top_k_reranked. If reranker is disabled or times out,
    returns the original list truncated to top_k_reranked by hybrid_score.
    """
    if cfg is None:
        cfg = RetrievalConfig()

    top_k = cfg.reranker.top_k_reranked

    if not cfg.reranker.enabled:
        log.debug("Reranker disabled — returning top-%d by hybrid score", top_k)
        return sorted(results, key=lambda r: r.hybrid_score, reverse=True)[:top_k]

    if not results:
        return []

    # Limit candidates passed to cross-encoder to control latency.
    # Cross-encoder scoring is O(n) — halving candidates halves inference time.
    candidates = sorted(results, key=lambda r: r.hybrid_score, reverse=True)
    candidates = candidates[: cfg.reranker.top_k_candidates]

    model_name = DEFAULT_MODEL
    model = _load_model(model_name)

    pairs = [(query, _compose_rerank_text(r)) for r in candidates]

    log.debug(
        "Reranking %d candidates (of %d) with %s (budget=%.0fms)",
        len(pairs), len(results), model_name, cfg.reranker.latency_budget_ms,
    )

    t0 = time.perf_counter()
    scores: list[float] = model.predict(pairs).tolist()  # type: ignore[union-attr]
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if elapsed_ms > cfg.reranker.latency_budget_ms:
        log.warning(
            "Reranker exceeded latency budget: %.0fms > %.0fms — "
            "returning original hybrid order. Consider disabling reranker.",
            elapsed_ms, cfg.reranker.latency_budget_ms,
        )
        return sorted(results, key=lambda r: r.hybrid_score, reverse=True)[:top_k]

    log.info(
        "Reranked %d candidates in %.0fms — keeping top %d",
        len(candidates), elapsed_ms, top_k,
    )

    reranked = []
    for result, score in zip(candidates, scores):
        updated = result.model_copy(update={"rerank_score": score, "final_score": score})
        reranked.append(updated)

    reranked.sort(key=lambda r: r.rerank_score or 0.0, reverse=True)
    return reranked[:top_k]


# ── CLI ───────────────────────────────────────────────────────────────────────
# TEST_QUERIES is imported from hybrid_search to keep the QA suite consistent
# across Phase 3b and 3c — results are directly comparable between phases.


def _print_comparison(
    query: str,
    before: list[SearchResult],
    after: list[SearchResult],
    verbose: bool = False,
) -> None:
    """Print side-by-side rank comparison: hybrid order vs reranked order."""
    before_ids = [r.chunk_id for r in before]

    print(f"\n  Query: {query!r}")

    rows = []
    for new_rank, r in enumerate(after, 1):
        old_rank = before_ids.index(r.chunk_id) + 1 if r.chunk_id in before_ids else -1
        movement = old_rank - new_rank
        move_str = f"↑{movement}" if movement > 0 else (f"↓{abs(movement)}" if movement < 0 else "—")
        rerank_score_str = f"{r.rerank_score:.4f}" if r.rerank_score is not None else "n/a"
        rows.append((new_rank, r.hybrid_score, rerank_score_str, move_str, r.chunk_id, _compose_rerank_text(r)))

    if verbose:
        for new_rank, hybrid, rerank_str, move_str, chunk_id, text in rows:
            print(f"\n  [{new_rank:02d}] hybrid={hybrid:.4f}  rerank={rerank_str}  move={move_str}  {chunk_id}")
            clean = text.replace("\n", " ")
            for i in range(0, len(clean), 100):
                print(f"       {clean[i:i+100]}")

    # Summary table always printed at the end
    print(f"\n  {'Rank':<6} {'Hybrid':<10} {'Rerank':<10} {'Move':<8} {'Text (first 60 chars)'}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*8} {'-'*60}")
    for new_rank, hybrid, rerank_str, move_str, chunk_id, text in rows:
        snippet = text.replace("\n", " ")[:60]
        print(f"  {new_rank:<6} {hybrid:<10.4f} {rerank_str:<10} {move_str:<8} {snippet!r}")


def run_test(cfg: RetrievalConfig) -> None:
    print("\n" + "=" * 70)
    print(f"reranker.py — QA test suite  [model={DEFAULT_MODEL}]")
    print(f"  reranker_enabled={cfg.reranker.enabled}  top_k_reranked={cfg.reranker.top_k_reranked}")
    print(f"  latency_budget_ms={cfg.reranker.latency_budget_ms}")
    print("=" * 70)

    voyage_key = os.environ.get("VOYAGE_API_KEY", "")
    if not voyage_key:
        raise EnvironmentError("VOYAGE_API_KEY not set")

    for query in TEST_QUERIES:
        print(f"\n[SEARCH] {query!r}")
        t0 = time.perf_counter()
        hybrid_results = hybrid_search(query, cfg=cfg, voyage_api_key=voyage_key)
        search_ms = (time.perf_counter() - t0) * 1000
        print(f"  Hybrid search: {len(hybrid_results)} results in {search_ms:.0f}ms")

        # Rerank
        t1 = time.perf_counter()
        reranked = rerank(query, hybrid_results, cfg=cfg)
        rerank_ms = (time.perf_counter() - t1) * 1000
        print(f"  Reranking: {len(reranked)} results in {rerank_ms:.0f}ms")

        # Compare against the full candidate window so movement reflects true hybrid rank
        _print_comparison(query, hybrid_results[:cfg.reranker.top_k_candidates], reranked, verbose=True)

    print("\n" + "=" * 70)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Phase 3c cross-encoder reranker")
    parser.add_argument("--query", type=str, help="Run a single query")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k reranked results (default from config)")
    parser.add_argument("--test", action="store_true", help="Run built-in QA test suite")
    parser.add_argument("--verbose", action="store_true", help="Print text previews")
    parser.add_argument(
        "--disable-reranker", action="store_true",
        help="Run with reranker disabled (toggle verification)"
    )
    args = parser.parse_args()

    cfg = RetrievalConfig()
    if args.top_k is not None:
        cfg.reranker.top_k_reranked = args.top_k
    cfg.reranker.enabled = not args.disable_reranker

    if args.test:
        run_test(cfg)
    elif args.query:
        voyage_key = os.environ.get("VOYAGE_API_KEY", "")
        if not voyage_key:
            raise EnvironmentError("VOYAGE_API_KEY not set")

        print("\n" + "=" * 70)
        print(f"reranker.py — single query  [model={DEFAULT_MODEL}]")
        print(f"  reranker_enabled={cfg.reranker.enabled}  top_k_reranked={cfg.reranker.top_k_reranked}")
        print("=" * 70)

        print(f"\n[SEARCH] {args.query!r}")
        t0 = time.perf_counter()
        hybrid_results = hybrid_search(args.query, cfg=cfg, voyage_api_key=voyage_key)
        search_ms = (time.perf_counter() - t0) * 1000
        print(f"  Hybrid search: {len(hybrid_results)} results in {search_ms:.0f}ms")

        t1 = time.perf_counter()
        reranked = rerank(args.query, hybrid_results, cfg=cfg)
        rerank_ms = (time.perf_counter() - t1) * 1000
        print(f"  Reranking: {len(reranked)} results in {rerank_ms:.0f}ms")

        _print_comparison(
            args.query,
            hybrid_results[:cfg.reranker.top_k_candidates],
            reranked,
            verbose=True,
        )
        print("\n" + "=" * 70)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
