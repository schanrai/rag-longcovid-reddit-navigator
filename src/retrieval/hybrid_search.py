"""
hybrid_search.py — BYOV query embedding + Weaviate hybrid search.

Phase 3b: two-step retrieval.

Step 1 — embed:  Call Voyage API with input_type="query" to get a query vector.
Step 2 — search: Call Weaviate hybrid() with both the rewritten text (BM25 leg)
                 and the query vector (semantic leg). Alpha controls the ratio.

⚠  BYOV requirement: LongCovidChunks has no native Weaviate vectorizer.
   The vector MUST be supplied explicitly or the semantic leg silently fails.

CLI usage:
    python -m src.retrieval.hybrid_search --query "fatigue 8 months post covid"
    python -m src.retrieval.hybrid_search --query "..." --top-k 20 --alpha 0.5
    python -m src.retrieval.hybrid_search --test
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Any, Final

import httpx
import weaviate
from dotenv import load_dotenv
from weaviate.classes.init import AdditionalConfig, Auth, Timeout
from weaviate.classes.query import HybridFusion, MetadataQuery

from .config import RetrievalConfig, SearchConfig
from .models import ChunkMetadata, SearchResult

load_dotenv()
log = logging.getLogger("retrieval.hybrid_search")

COLLECTION_NAME: Final[str] = "LongCovidChunks"

# Properties to return from Weaviate — must match the stored schema
RETURN_PROPERTIES: Final[list[str]] = [
    "chunk_id",
    "text",
    "chunk_type",
    "post_title",
    "post_summary",
    "permalink",
    "created_utc",
    "comment_score",
    "post_score",
    "agreement_count",
    "thanks_count",
    "num_comments",
    "upvote_ratio",
    "nest_level",
    "is_submitter",
    "stickied",
    "chunk_index",
    "total_chunks",
    "word_count",
    "link_flair_text",
    "link_id",
    "parent_id",
]


# ── Step 1: Voyage query embedding ────────────────────────────────────────────

def embed_query(
    query: str,
    *,
    cfg: SearchConfig,
    api_key: str,
) -> list[float]:
    """
    Embed a single query string via Voyage API.

    Uses input_type='query' — Voyage applies asymmetric encoding that optimises
    query vectors for retrieval against document vectors.
    """
    payload: dict[str, Any] = {
        "model": cfg.voyage_model,
        "input": [query],
        "input_type": "query",
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    log.debug("Embedding query via Voyage: model=%s query=%r", cfg.voyage_model, query)

    with httpx.Client(timeout=cfg.voyage_timeout_s) as http:
        r = http.post(cfg.voyage_embed_url, headers=headers, json=payload)
        r.raise_for_status()
        body = r.json()

    data = body.get("data")
    if not data or len(data) == 0:
        raise ValueError(f"Voyage returned empty data for query: {query!r}")

    embedding: list[float] = data[0]["embedding"]
    log.debug("Query embedded: %d dimensions", len(embedding))
    return embedding


# ── Step 2: Weaviate hybrid search ────────────────────────────────────────────

def _build_weaviate_client() -> weaviate.WeaviateClient:
    """Construct a Weaviate Cloud client from environment variables."""
    url = os.environ.get("WEAVIATE_URL", "")
    api_key = os.environ.get("WEAVIATE_API_KEY", "")
    if not url or not api_key:
        raise EnvironmentError(
            "WEAVIATE_URL and WEAVIATE_API_KEY must be set in the environment."
        )
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=url,
        auth_credentials=Auth.api_key(api_key),
        additional_config=AdditionalConfig(
            timeout=Timeout(init=60, query=60, insert=120)
        ),
        skip_init_checks=True,
    )


def _parse_weaviate_object(obj: Any, hybrid_score: float) -> SearchResult:
    """Convert a Weaviate object into a SearchResult."""
    props = obj.properties

    # Use `or` fallbacks throughout — Weaviate may return explicit None for
    # optional fields even when the key is present in the response object.
    metadata = ChunkMetadata(
        chunk_id=props.get("chunk_id") or "",
        chunk_type=props.get("chunk_type") or "",
        post_title=props.get("post_title") or "",
        post_summary=props.get("post_summary") or "",
        permalink=props.get("permalink") or "",
        created_utc=props.get("created_utc"),
        comment_score=props.get("comment_score"),
        post_score=props.get("post_score"),
        agreement_count=int(props.get("agreement_count") or 0),
        thanks_count=int(props.get("thanks_count") or 0),
        num_comments=props.get("num_comments"),
        upvote_ratio=props.get("upvote_ratio"),
        nest_level=props.get("nest_level"),
        is_submitter=bool(props.get("is_submitter") or False),
        stickied=bool(props.get("stickied") or False),
        chunk_index=int(props.get("chunk_index") or 0),
        total_chunks=int(props.get("total_chunks") or 1),
        word_count=int(props.get("word_count") or 0),
        link_flair_text=props.get("link_flair_text"),
        link_id=props.get("link_id"),
        parent_id=props.get("parent_id"),
    )

    return SearchResult(
        chunk_id=metadata.chunk_id,
        text=props.get("text", ""),
        hybrid_score=hybrid_score,
        final_score=hybrid_score,  # overwritten by ranking.py
        metadata=metadata,
    )


def weaviate_hybrid_search(
    query_text: str,
    query_vector: list[float],
    *,
    cfg: SearchConfig,
    client: weaviate.WeaviateClient,
) -> list[SearchResult]:
    """
    Run a hybrid search against the LongCovidChunks collection.

    Both query_text (BM25 leg) and query_vector (semantic leg) are required.
    Weaviate performs RRF fusion internally; alpha controls the weighting.
    """
    log.debug(
        "Weaviate hybrid search: alpha=%.2f top_k=%d query=%r",
        cfg.alpha, cfg.top_k_initial, query_text,
    )

    collection = client.collections.get(COLLECTION_NAME)

    response = collection.query.hybrid(
        query=query_text,
        vector=query_vector,
        alpha=cfg.alpha,
        limit=cfg.top_k_initial,
        query_properties=["text", "post_title"],
        return_properties=RETURN_PROPERTIES,
        return_metadata=MetadataQuery(score=True),
        fusion_type=HybridFusion.RELATIVE_SCORE,
    )

    results: list[SearchResult] = []
    for obj in response.objects:
        score = obj.metadata.score if obj.metadata and obj.metadata.score else 0.0
        results.append(_parse_weaviate_object(obj, hybrid_score=score))

    log.info(
        "Hybrid search returned %d results for query=%r (alpha=%.2f)",
        len(results), query_text, cfg.alpha,
    )
    return results


# ── Deduplication ─────────────────────────────────────────────────────────────

def dedup_results(
    results: list[SearchResult],
    *,
    top_k: int,
) -> list[SearchResult]:
    """
    Remove exact-text duplicate chunks, keeping the first (highest-scoring) copy.

    The corpus contains ~945 duplicate chunks (561 groups) — mostly the same user
    posting identical comments across multiple threads. Without dedup, a single
    duplicate can consume two or more retrieval slots, reducing result diversity.

    Deduplication is done by exact text hash. Near-duplicate detection (e.g. fuzzy
    matching) is out of scope for this phase.

    Parameters
    ----------
    results:
        Candidates from weaviate_hybrid_search(), sorted by hybrid_score descending.
    top_k:
        Maximum unique results to return. Truncates after dedup.

    Returns
    -------
    list[SearchResult] with duplicates removed, truncated to top_k.
    """
    import hashlib

    seen: set[str] = set()
    deduped: list[SearchResult] = []

    for r in results:
        text_hash = hashlib.md5(r.text.strip().encode()).hexdigest()
        if text_hash in seen:
            continue
        seen.add(text_hash)
        deduped.append(r)
        if len(deduped) >= top_k:
            break

    removed = len(results) - len(deduped)
    if removed:
        log.info("Dedup removed %d duplicate chunk(s) — %d unique results", removed, len(deduped))

    return deduped


# ── Public entry point ────────────────────────────────────────────────────────

def search(
    query_text: str,
    *,
    cfg: RetrievalConfig | None = None,
    voyage_api_key: str | None = None,
    weaviate_client: weaviate.WeaviateClient | None = None,
) -> list[SearchResult]:
    """
    Full Phase 3b search: embed query then run hybrid search.

    Parameters
    ----------
    query_text:
        The rewritten query string from Phase 3a.
    cfg:
        RetrievalConfig; defaults to production defaults if not provided.
    voyage_api_key:
        Voyage API key; falls back to VOYAGE_API_KEY env var.
    weaviate_client:
        Pre-built Weaviate client; if not provided, one is constructed from env vars.
        Caller is responsible for closing the client when done.

    Returns
    -------
    list[SearchResult] sorted by hybrid_score descending.
    """
    if cfg is None:
        cfg = RetrievalConfig()

    api_key = voyage_api_key or os.environ.get("VOYAGE_API_KEY", "")
    if not api_key:
        raise EnvironmentError("VOYAGE_API_KEY must be set in the environment.")

    own_client = weaviate_client is None
    client = weaviate_client or _build_weaviate_client()

    try:
        t0 = time.perf_counter()
        query_vector = embed_query(query_text, cfg=cfg.search, api_key=api_key)
        raw_results = weaviate_hybrid_search(
            query_text, query_vector, cfg=cfg.search, client=client
        )
        results = dedup_results(raw_results, top_k=cfg.search.top_k_deduped)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        log.info("search() completed in %.0fms — %d results", elapsed_ms, len(results))
        return results
    finally:
        if own_client:
            client.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

TEST_QUERIES_CORPUS: Final[list[str]] = [
    # ── Corpus-derived queries (from earlier QA phases, retained for cross-phase comparability) ──
    # From hybrid_search_qa.md Test 1 — timeline fatigue query, surface-level semantic match
    "Long COVID persistent fatigue 8 months after infection",
    # From hybrid_search_qa.md Test 4 — fuzzy semantic, no medical keywords (critical test for vector leg)
    "I feel like my brain stopped working",
    # From query_rewrite_singlellm_qa.md q09 — community signal + informal phrasing; revealed stripping issue in 3a
    "beta blockers for the constant tachycardia thing — anyone? ivabradine?",
    # From query_rewrite_singlellm_qa.md q12 — emotional framing + dual intent; revealed misclassification in 3a
    "LC awareness. Doctor and family don't believe me =(",
]

TEST_QUERIES_SYNTHETIC: Final[list[str]] = [
    # ── Synthetic user-style queries (diversity test — should not cluster on a single thread) ──
    "why am i still so exhausted 8 months after getting Covid? Is this normal??",
    "Everything feels foggy and I can't think straight anymore",
    "Anyone taking beta blockers for tachycardia-like symptoms? has it helped? what about ivabradine?",
    "How do I get my doctor to take my symptoms seriously?",
]

TEST_QUERIES: Final[list[str]] = [
    TEST_QUERIES_SYNTHETIC[0],  # Q5 — exhaustion / 8 months
    TEST_QUERIES_SYNTHETIC[3],  # Q8 — doctor credibility
]


def _print_results(results: list[SearchResult], verbose: bool = False) -> None:
    for i, r in enumerate(results, 1):
        m = r.metadata
        score_str = f"{r.hybrid_score:.4f}"
        chunk_type = m.chunk_type or "?"
        print(
            f"  [{i:02d}] score={score_str}  type={chunk_type:<8}  "
            f"agree={m.agreement_count}  thanks={m.thanks_count}  "
            f"permalink={m.permalink or 'n/a'}"
        )
        if verbose:
            preview = r.text[:200].replace("\n", " ")
            print(f"       {preview!r}")


def run_test(cfg: RetrievalConfig) -> None:
    print("\n" + "=" * 70)
    print(f"hybrid_search.py — QA test suite  [alpha={cfg.search.alpha}  top_k={cfg.search.top_k_initial}]")
    print("=" * 70)

    client = _build_weaviate_client()
    voyage_key = os.environ.get("VOYAGE_API_KEY", "")
    if not voyage_key:
        raise EnvironmentError("VOYAGE_API_KEY not set")

    try:
        for query in TEST_QUERIES:
            print(f"\nQuery: {query!r}")
            t0 = time.perf_counter()
            vector = embed_query(query, cfg=cfg.search, api_key=voyage_key)
            results = weaviate_hybrid_search(query, vector, cfg=cfg.search, client=client)
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"  → {len(results)} results in {elapsed:.0f}ms")
            _print_results(results[:5])
    finally:
        client.close()

    print("\n" + "=" * 70)
    print(f"Results: {len(TEST_QUERIES)} queries / {cfg.search.top_k_initial} candidates each")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Phase 3b hybrid search")
    parser.add_argument("--query", type=str, help="Run a single query")
    parser.add_argument("--top-k", type=int, default=50, help="Number of results (default 50)")
    parser.add_argument("--alpha", type=float, default=0.7, help="BM25/vector ratio (default 0.7)")
    parser.add_argument("--test", action="store_true", help="Run built-in QA test suite")
    parser.add_argument("--verbose", action="store_true", help="Print text previews")
    args = parser.parse_args()

    cfg = RetrievalConfig()
    cfg.search.top_k_initial = args.top_k
    cfg.search.alpha = args.alpha

    if args.test:
        run_test(cfg)
    elif args.query:
        print(f"\nQuery: {args.query!r}")
        t0 = time.perf_counter()
        results = search(args.query, cfg=cfg)
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"→ {len(results)} results in {elapsed:.0f}ms\n")
        _print_results(results, verbose=args.verbose)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
