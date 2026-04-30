#!/usr/bin/env python3
"""
index_weaviate.py — Phase 2: embed full corpus with Voyage-4-Large and ingest into Weaviate Cloud.

Reads:
  data/comment_chunks.jsonl   — enriched comment chunks (164,812 records)
  data/post_chunks.jsonl      — post chunks (19,443 records)

Writes:
  reports/index_report.json   — ingest summary (counts, errors, timing)

Collection schema: LongCovidChunks
  - Vectors: pre-computed via Voyage API (bring-your-own-vector; atom-storage
    composition done here, consistent with embed_eval.py / compose_embedding_input)
  - BM25 (keyword): enabled on text + post_title properties (Weaviate default)
  - All chunk metadata stored as properties for hybrid search + ranking

Usage:
  python3 src/index_weaviate.py                      # full corpus
  python3 src/index_weaviate.py --dry-run            # schema only, no data
  python3 src/index_weaviate.py --limit 500          # smoke test: first N chunks
  python3 src/index_weaviate.py --skip-comments      # post chunks only
  python3 src/index_weaviate.py --skip-posts         # comment chunks only
  python3 src/index_weaviate.py --recreate           # drop + recreate collection first
  python3 src/index_weaviate.py --enrichment-policy depth_aware_v1   # match original depth-aware eval gate
  python3 src/index_weaviate.py --enrichment-policy depth_aware_v2   # depth3+ parent_first_sentence+body
  python3 src/index_weaviate.py --enrichment-policy depth_aware_blend  # depth3+ title+parent_first_sentence+body

Long runs: run in a standalone terminal (not IDE) to avoid OOM/timeout issues:
  python3 -u src/index_weaviate.py 2>&1 | tee reports/index_weaviate_run.log
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import httpx
import weaviate
from dotenv import load_dotenv
from weaviate.classes.config import DataType, Property
from weaviate.classes.init import Auth, AdditionalConfig, Timeout

load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent))
from embed_eval import EnrichmentPolicy, compose_embedding_input

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("index_weaviate")

# ── Constants ──────────────────────────────────────────────────────────────────

COLLECTION_NAME: Final[str] = "LongCovidChunks"
VOYAGE_MODEL: Final[str] = "voyage-4-large"
VOYAGE_EMBED_URL: Final[str] = "https://api.voyageai.com/v1/embeddings"
VOYAGE_BATCH_MAX: Final[int] = 64
WEAVIATE_BATCH_SIZE: Final[int] = 200
PROGRESS_INTERVAL: Final[int] = 1_000


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Config:
    data_dir: Path = Path(__file__).parent.parent / "data"
    reports_dir: Path = Path(__file__).parent.parent / "reports"
    comment_chunks_file: str = "comment_chunks.jsonl"
    post_chunks_file: str = "post_chunks.jsonl"
    report_out: str = "index_report.json"


# ── Weaviate collection schema ─────────────────────────────────────────────────

def create_collection(client: weaviate.WeaviateClient) -> None:
    """
    Create the LongCovidChunks collection with explicit property schema.

    BM25 is enabled by default on all text properties in Weaviate — no extra
    configuration required. Vectors are provided at import time (bring-your-own-
    vector) so no vectorizer is configured here.

    Vector index: HNSW with RQ 8-bit compression (set at cluster creation time).
    If the cluster was created without compression, vectors remain uncompressed —
    this schema is compression-agnostic.
    """
    client.collections.create(
        name=COLLECTION_NAME,
        description="Long COVID Reddit chunks — comments and posts, enriched with post context",
        properties=[
            # ── Core content ──────────────────────────────────────────────────
            Property(name="chunk_id",    data_type=DataType.TEXT,   skip_vectorization=True),
            Property(name="text",        data_type=DataType.TEXT),
            Property(name="post_title",  data_type=DataType.TEXT),
            Property(name="post_summary",data_type=DataType.TEXT),
            Property(name="chunk_type",  data_type=DataType.TEXT,   skip_vectorization=True),  # "comment" | "post"

            # ── Ranking signals ───────────────────────────────────────────────
            Property(name="comment_score",   data_type=DataType.INT,   skip_vectorization=True),
            Property(name="post_score",      data_type=DataType.INT,   skip_vectorization=True),
            Property(name="agreement_count", data_type=DataType.INT,   skip_vectorization=True),
            Property(name="thanks_count",    data_type=DataType.INT,   skip_vectorization=True),
            Property(name="num_comments",    data_type=DataType.INT,   skip_vectorization=True),
            Property(name="upvote_ratio",    data_type=DataType.NUMBER,skip_vectorization=True),

            # ── Citation / provenance ─────────────────────────────────────────
            Property(name="permalink",       data_type=DataType.TEXT,  skip_vectorization=True),
            Property(name="link_id",         data_type=DataType.TEXT,  skip_vectorization=True),
            Property(name="parent_id",       data_type=DataType.TEXT,  skip_vectorization=True),
            Property(name="link_flair_text", data_type=DataType.TEXT,  skip_vectorization=True),

            # ── Thread metadata ───────────────────────────────────────────────
            Property(name="nest_level",  data_type=DataType.INT,  skip_vectorization=True),
            Property(name="parent_first_sentence", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="is_submitter",data_type=DataType.BOOL, skip_vectorization=True),
            Property(name="stickied",    data_type=DataType.BOOL, skip_vectorization=True),
            Property(name="created_utc", data_type=DataType.INT,  skip_vectorization=True),

            # ── Chunk housekeeping ────────────────────────────────────────────
            Property(name="chunk_index",  data_type=DataType.INT, skip_vectorization=True),
            Property(name="total_chunks", data_type=DataType.INT, skip_vectorization=True),
            Property(name="word_count",   data_type=DataType.INT, skip_vectorization=True),
        ],
        # No vectorizer: vectors are provided at import time via batch.add_object(vector=...)
        # HNSW is the Weaviate default — no vector_index_config needed.
        # RQ 8-bit compression was configured at cluster creation time.
    )
    log.info("Collection '%s' created", COLLECTION_NAME)


# ── Voyage embedding ───────────────────────────────────────────────────────────

def voyage_embed_batch(
    texts: list[str],
    *,
    api_key: str,
    input_type: str = "document",
    batch_size: int = VOYAGE_BATCH_MAX,
    max_retries: int = 5,
) -> list[list[float]]:
    """
    Embed a list of texts via Voyage API. Returns list of float vectors.
    Same retry/backoff pattern as embed_eval.py.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    out: list[list[float]] = []
    n = len(texts)

    with httpx.Client(timeout=120.0) as http:
        for start in range(0, n, batch_size):
            batch = texts[start : start + batch_size]
            payload: dict[str, Any] = {
                "model": VOYAGE_MODEL,
                "input": batch,
                "input_type": input_type,
            }
            for attempt in range(max_retries):
                try:
                    r = http.post(VOYAGE_EMBED_URL, headers=headers, json=payload)
                    r.raise_for_status()
                    body = r.json()
                    data = body.get("data")
                    if not data or len(data) != len(batch):
                        raise ValueError("Voyage response data length mismatch")
                    data_sorted = sorted(data, key=lambda x: int(x.get("index", 0)))
                    for row in data_sorted:
                        emb = row.get("embedding")
                        if not emb:
                            raise ValueError("Missing embedding in Voyage row")
                        out.append(emb)
                    break
                except (httpx.HTTPError, ValueError) as exc:
                    wait = 2.0 * (attempt + 1)
                    log.warning(
                        "Voyage batch offset=%s attempt=%s failed (%s), retry in %.0fs",
                        start, attempt + 1, exc, wait,
                    )
                    time.sleep(wait)
            else:
                raise RuntimeError(f"Voyage embedding failed permanently at offset {start}")

    return out


# ── Chunk record → Weaviate properties ────────────────────────────────────────

def chunk_to_properties(rec: dict[str, Any], chunk_type: str) -> dict[str, Any]:
    """
    Map a chunk record to the Weaviate property dict.
    None values are omitted — Weaviate treats missing properties as null.
    """
    props: dict[str, Any] = {
        "chunk_id":    str(rec.get("chunk_id") or "").strip(),
        "text":        str(rec.get("text") or ""),
        "post_title":  str(rec.get("post_title") or ""),
        "chunk_type":  chunk_type,
        "chunk_index":  int(rec.get("chunk_index") or 0),
        "total_chunks": int(rec.get("total_chunks") or 1),
        "word_count":   int(rec.get("word_count") or 0),
        "agreement_count": int(rec.get("agreement_count") or 0),
        "thanks_count":    int(rec.get("thanks_count") or 0),
        "stickied":   bool(rec.get("stickied", False)),
    }

    # Optional fields — only include if non-null
    if rec.get("post_summary"):
        props["post_summary"] = str(rec["post_summary"])
    if rec.get("created_utc") is not None:
        props["created_utc"] = int(rec["created_utc"])

    if chunk_type == "comment":
        props["comment_score"] = int(rec.get("comment_score") or 0)
        props["link_id"]       = str(rec.get("link_id") or "")
        props["parent_id"]     = str(rec.get("parent_id") or "")
        props["nest_level"]    = int(rec.get("nest_level") or 0)
        props["is_submitter"]  = bool(rec.get("is_submitter", False))
        # Observability contract: store parent sentence only where v2 policy uses it.
        # For depth >= 3 comments we persist the parent sentence when available;
        # otherwise leave the property unset (null/missing in Weaviate).
        if props["nest_level"] >= 3:
            parent_first = (rec.get("parent_first_sentence") or "").strip()
            if parent_first:
                props["parent_first_sentence"] = parent_first
    else:
        props["post_score"]      = int(rec.get("post_score") or 0)
        props["num_comments"]    = int(rec.get("num_comments") or 0)
        props["permalink"]       = str(rec.get("permalink") or "")
        if rec.get("upvote_ratio") is not None:
            props["upvote_ratio"] = float(rec["upvote_ratio"])
        if rec.get("link_flair_text"):
            props["link_flair_text"] = str(rec["link_flair_text"])

    return props


# ── Stream + ingest ────────────────────────────────────────────────────────────

@dataclass
class IngestStats:
    indexed: int = 0
    failed: int = 0
    skipped: int = 0
    voyage_calls: int = 0
    elapsed_s: float = 0.0
    errors: list[str] = field(default_factory=list)


def ingest_jsonl(
    path: Path,
    chunk_type: str,
    collection: Any,
    voyage_key: str,
    *,
    enrichment_policy: EnrichmentPolicy = "baseline_full",
    limit: int | None = None,
    weaviate_batch_size: int = WEAVIATE_BATCH_SIZE,
    voyage_batch_size: int = VOYAGE_BATCH_MAX,
) -> IngestStats:
    """
    Stream a chunk JSONL file, embed with Voyage in batches, and upsert into Weaviate.

    Strategy: accumulate records up to weaviate_batch_size, embed the whole
    micro-batch together (one or more Voyage calls), then send to Weaviate.
    This keeps Voyage batches full and Weaviate batches efficient.
    """
    stats = IngestStats()
    t0 = time.monotonic()
    pending_recs: list[dict[str, Any]] = []

    def flush(batch: list[dict[str, Any]]) -> None:
        if not batch:
            return
        texts = [
            compose_embedding_input(r, enrichment_policy=enrichment_policy)
            for r in batch
        ]
        vectors = voyage_embed_batch(
            texts,
            api_key=voyage_key,
            input_type="document",
            batch_size=voyage_batch_size,
        )
        stats.voyage_calls += (len(texts) + voyage_batch_size - 1) // voyage_batch_size

        with collection.batch.fixed_size(batch_size=len(batch)) as wb:
            for rec, vec in zip(batch, vectors):
                props = chunk_to_properties(rec, chunk_type)
                wb.add_object(properties=props, vector=vec)

        # Check for Weaviate batch errors
        failed = collection.batch.failed_objects
        if failed:
            for fo in failed:
                props = getattr(fo.object_, "properties", None) or {}
                cid = props.get("chunk_id", fo.original_uuid)
                msg = f"chunk_id={cid} err={fo.message}"
                stats.errors.append(msg)
                log.warning("Weaviate ingest error: %s", msg)
            stats.failed += len(failed)
        stats.indexed += len(batch) - len(failed)

    log.info(
        "=== Ingesting %s from %s (enrichment_policy=%s) ===",
        chunk_type,
        path.name,
        enrichment_policy,
    )
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                log.warning("JSON parse error: %s", exc)
                stats.skipped += 1
                continue

            pending_recs.append(rec)

            if len(pending_recs) >= weaviate_batch_size:
                flush(pending_recs)
                pending_recs.clear()
                total = stats.indexed + stats.failed
                if total % PROGRESS_INTERVAL < weaviate_batch_size:
                    log.info(
                        "  %s: %s indexed, %s failed so far",
                        chunk_type, stats.indexed, stats.failed,
                    )

            if limit and (stats.indexed + stats.failed + stats.skipped) >= limit:
                log.info("  --limit %s reached, stopping early", limit)
                break

    flush(pending_recs)  # final partial batch
    stats.elapsed_s = time.monotonic() - t0
    log.info(
        "  %s done — indexed=%s failed=%s skipped=%s voyage_calls=%s time=%.1fs",
        chunk_type, stats.indexed, stats.failed, stats.skipped,
        stats.voyage_calls, stats.elapsed_s,
    )
    return stats


# ── Report ─────────────────────────────────────────────────────────────────────

def build_report(
    comment_stats: IngestStats | None,
    post_stats: IngestStats | None,
    collection_name: str,
    *,
    enrichment_policy: EnrichmentPolicy,
) -> dict[str, Any]:
    return {
        "schema": "index_report_v1",
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "collection": collection_name,
        "enrichment_policy": enrichment_policy,
        "voyage_model": VOYAGE_MODEL,
        "comment_chunks": {
            "indexed": comment_stats.indexed if comment_stats else 0,
            "failed":  comment_stats.failed  if comment_stats else 0,
            "skipped": comment_stats.skipped if comment_stats else 0,
            "voyage_calls": comment_stats.voyage_calls if comment_stats else 0,
            "elapsed_s": round(comment_stats.elapsed_s, 1) if comment_stats else 0,
            "errors": (comment_stats.errors[:20] if comment_stats else []),
        },
        "post_chunks": {
            "indexed": post_stats.indexed if post_stats else 0,
            "failed":  post_stats.failed  if post_stats else 0,
            "skipped": post_stats.skipped if post_stats else 0,
            "voyage_calls": post_stats.voyage_calls if post_stats else 0,
            "elapsed_s": round(post_stats.elapsed_s, 1) if post_stats else 0,
            "errors": (post_stats.errors[:20] if post_stats else []),
        },
        "totals": {
            "indexed": (comment_stats.indexed if comment_stats else 0)
                      + (post_stats.indexed if post_stats else 0),
            "failed":  (comment_stats.failed  if comment_stats else 0)
                      + (post_stats.failed  if post_stats else 0),
        },
    }


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = Config()
    ap = argparse.ArgumentParser(description="Embed + ingest full corpus into Weaviate Cloud.")
    ap.add_argument("--dry-run",       action="store_true", help="Create schema only, no data")
    ap.add_argument("--recreate",      action="store_true", help="Drop + recreate collection")
    ap.add_argument("--skip-comments", action="store_true", help="Skip comment chunks")
    ap.add_argument("--skip-posts",    action="store_true", help="Skip post chunks")
    ap.add_argument("--limit",         type=int, default=None, help="Max chunks per file (smoke test)")
    ap.add_argument("--weaviate-batch",type=int, default=WEAVIATE_BATCH_SIZE)
    ap.add_argument("--voyage-batch",  type=int, default=VOYAGE_BATCH_MAX)
    ap.add_argument(
        "--enrichment-policy",
        choices=("baseline_full", "no_enrich", "depth_aware_v1", "depth_aware_v2", "depth_aware_blend"),
        default="baseline_full",
        help="Embedding input composition (same as embed_eval.compose_embedding_input). "
        "Use depth_aware_v1, depth_aware_v2, or depth_aware_blend for reconstructed "
        "nest_level + parent_first_sentence corpus.",
    )
    ap.add_argument(
        "--out", type=Path,
        default=cfg.reports_dir / cfg.report_out,
    )
    args = ap.parse_args()
    enrichment_policy: EnrichmentPolicy = args.enrichment_policy

    voyage_key = os.environ.get("VOYAGE_API_KEY", "").strip()
    weaviate_url = os.environ.get("WEAVIATE_URL", "").strip()
    weaviate_key = os.environ.get("WEAVIATE_API_KEY", "").strip()

    for name, val in [
        ("VOYAGE_API_KEY", voyage_key),
        ("WEAVIATE_URL", weaviate_url),
        ("WEAVIATE_API_KEY", weaviate_key),
    ]:
        if not val:
            log.error("%s not set in environment", name)
            sys.exit(1)

    comment_path = cfg.data_dir / cfg.comment_chunks_file
    post_path    = cfg.data_dir / cfg.post_chunks_file
    for p in (comment_path, post_path):
        if not p.exists():
            log.error("Missing file: %s", p)
            sys.exit(1)

    log.info(
        "Connecting to Weaviate Cloud: %s (enrichment_policy=%s)",
        weaviate_url,
        enrichment_policy,
    )
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_key),
        additional_config=AdditionalConfig(timeout=Timeout(init=60, query=120, insert=300)),
        skip_init_checks=True,
    )

    try:
        exists = client.collections.exists(COLLECTION_NAME)

        if args.recreate and exists:
            log.info("--recreate: dropping existing collection '%s'", COLLECTION_NAME)
            client.collections.delete(COLLECTION_NAME)
            exists = False

        if not exists:
            create_collection(client)
        else:
            log.info("Collection '%s' already exists — appending data", COLLECTION_NAME)

        if args.dry_run:
            log.info("--dry-run: schema ready, skipping data ingest")
            return

        collection = client.collections.get(COLLECTION_NAME)
        comment_stats: IngestStats | None = None
        post_stats: IngestStats | None = None

        if not args.skip_comments:
            comment_stats = ingest_jsonl(
                comment_path, "comment", collection, voyage_key,
                enrichment_policy=enrichment_policy,
                limit=args.limit,
                weaviate_batch_size=args.weaviate_batch,
                voyage_batch_size=args.voyage_batch,
            )

        if not args.skip_posts:
            post_stats = ingest_jsonl(
                post_path, "post", collection, voyage_key,
                enrichment_policy=enrichment_policy,
                limit=args.limit,
                weaviate_batch_size=args.weaviate_batch,
                voyage_batch_size=args.voyage_batch,
            )

        report = build_report(
            comment_stats,
            post_stats,
            COLLECTION_NAME,
            enrichment_policy=enrichment_policy,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("Wrote index report → %s", args.out)

        totals = report["totals"]
        log.info(
            "✅  Total indexed: %s | failed: %s",
            totals["indexed"], totals["failed"],
        )

    finally:
        client.close()


if __name__ == "__main__":
    main()
