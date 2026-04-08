#!/usr/bin/env python3
"""
build_eval_corpus.py — Phase 1b: assemble eval corpus for embedding comparison.

Auto-selects positives from the same candidate ranking as suggest_eval_chunks.py,
dedupes and caps at ~25% of corpus, fills remaining ~75% with stratified random
distractors (not in the positive union).

Writes:
  data/eval_corpus.jsonl           — full chunk records (one JSON object per line)
  data/eval_corpus_meta.json       — build stats, seeds, paths
  data/eval_corpus_positives.json  — query_id → candidate positive chunk_ids (pre-dedupe lists)

Prerequisite: run suggest_eval_chunks.py first if you want an up-to-date report
(optional — this script re-scans using the same logic).

Usage:
  cd projects/rag-longcovid-reddit-navigator && python3 src/build_eval_corpus.py
  python3 src/build_eval_corpus.py --target-total 2000 --positive-cap 500 --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent))
from suggest_eval_chunks import CandidateRow, Config as SuggestConfig, resolve_top_candidates, stream_chunks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("build_eval_corpus")

SCHEMA_VERSION: Final[str] = "eval_corpus_meta_v1"
POSITIVE_FRACTION: Final[float] = 0.25


class EvalCorpusMeta(BaseModel):
    """Serialized build manifest for QA and reproducibility."""

    schema_version: str = Field(default=SCHEMA_VERSION)
    target_total: int
    positive_cap: int
    positive_fraction: float
    seed: int
    unique_positives_in_corpus: int
    distractor_count: int
    total_lines_written: int
    comment_chunks_source: str
    post_chunks_source: str
    suggest_max_per_query: int
    query_id_to_candidate_positives: dict[str, list[str]] = Field(default_factory=dict)


def _vote_bucket(rec: dict[str, Any]) -> str:
    v = int(rec.get("comment_score") or rec.get("post_score") or 0)
    if v <= 0:
        return "low"
    if v <= 10:
        return "mid"
    return "high"


def _chunk_kind(rec: dict[str, Any]) -> str:
    return "comment" if rec.get("comment_id") else "post"


def _year_bucket(rec: dict[str, Any]) -> str:
    ts = rec.get("created_utc")
    if ts is None:
        return "unknown"
    try:
        y = int(float(ts))
        from datetime import datetime, timezone

        dt = datetime.fromtimestamp(y, tz=timezone.utc)
        return str(dt.year)
    except (OSError, ValueError, OverflowError):
        return "unknown"


def _stratum_key(rec: dict[str, Any]) -> str:
    return f"{_chunk_kind(rec)}|{_vote_bucket(rec)}|{_year_bucket(rec)}"


def _merge_positive_chunk_ids(resolved: dict[str, list[CandidateRow]], cap: int) -> list[str]:
    """Global dedupe by descending best score across all queries."""
    rows: list[tuple[int, str]] = []
    for qid, lst in resolved.items():
        for r in lst:
            rows.append((r.score, r.chunk_id))
    rows.sort(key=lambda x: (-x[0], x[1]))
    out: list[str] = []
    seen: set[str] = set()
    for _score, cid in rows:
        if cid in seen:
            continue
        seen.add(cid)
        out.append(cid)
        if len(out) >= cap:
            break
    return out


def _query_id_to_candidates(resolved: dict[str, list[CandidateRow]]) -> dict[str, list[str]]:
    return {qid: [r.chunk_id for r in lst] for qid, lst in resolved.items()}


def _load_chunk_index(
    comment_path: Path,
    post_path: Path,
) -> dict[str, dict[str, Any]]:
    """chunk_id -> full record."""
    index: dict[str, dict[str, Any]] = {}
    for path in (comment_path, post_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing chunk file: {path}")
        for rec in stream_chunks(path):
            cid = rec.get("chunk_id")
            if not cid:
                continue
            index[str(cid)] = rec
    return index


def _stratified_sample_distractors(
    *,
    pool_by_stratum: dict[str, list[str]],
    n_need: int,
    rng: random.Random,
) -> list[str]:
    """Sample without replacement, proportional to stratum sizes (largest remainder)."""
    strata = {k: v[:] for k, v in pool_by_stratum.items() if v}
    total = sum(len(v) for v in strata.values())
    if total < n_need:
        raise ValueError(
            f"Not enough distractor candidates: need {n_need}, pool has {total}. "
            "Lower positive cap or target_total."
        )

    keys = sorted(strata.keys())
    weights = [len(strata[k]) for k in keys]
    raw = [n_need * w / total for w in weights]
    floors = [int(x) for x in raw]
    rem = n_need - sum(floors)
    frac_idx = sorted(
        range(len(keys)),
        key=lambda i: (raw[i] - floors[i], keys[i]),
        reverse=True,
    )
    for j in range(max(0, rem)):
        floors[frac_idx[j % len(frac_idx)]] += 1
    targets = dict(zip(keys, floors, strict=True))

    chosen: list[str] = []
    for key in keys:
        k = targets[key]
        if k <= 0:
            continue
        bucket = strata[key]
        if k > len(bucket):
            raise ValueError(f"Stratum {key!r}: need {k} samples but only {len(bucket)} available")
        chosen.extend(rng.sample(bucket, k=k))
    rng.shuffle(chosen)
    return chosen


@dataclass(frozen=True)
class BuildConfig:
    root: Path
    golden_path: Path
    comment_chunks: Path
    post_chunks: Path
    suggest_comment_chunks: Path
    suggest_post_chunks: Path
    out_corpus: Path
    out_meta: Path
    out_positives: Path
    target_total: int
    positive_cap: int
    suggest_max_per_query: int
    text_preview_len: int
    seed: int


def run(cfg: BuildConfig) -> EvalCorpusMeta:
    rng = random.Random(cfg.seed)

    suggest_cfg = SuggestConfig(
        data_dir=cfg.root / "data",
        golden_path=cfg.golden_path,
        comment_chunks=cfg.suggest_comment_chunks,
        post_chunks=cfg.suggest_post_chunks,
        out_path=cfg.root / "reports" / "eval_candidate_report.md",
        max_per_query=cfg.suggest_max_per_query,
        text_preview_len=cfg.text_preview_len,
    )
    log.info("Resolving top candidates (same logic as suggest_eval_chunks.py)…")
    resolved = resolve_top_candidates(suggest_cfg)
    query_pos_map = _query_id_to_candidates(resolved)

    pos_cap = min(cfg.positive_cap, max(1, int(cfg.target_total * POSITIVE_FRACTION)))
    positive_ids = _merge_positive_chunk_ids(resolved, cap=pos_cap)
    positive_set = set(positive_ids)

    n_pos = len(positive_ids)
    target_total = cfg.target_total
    n_neg = target_total - n_pos
    if n_neg < 0:
        raise ValueError(
            f"positive_cap ({pos_cap}) exceeds target_total ({target_total}). "
            "Increase --target-total or lower --positive-cap."
        )

    log.info("Loading chunk records for output + distractor pool…")
    chunk_index = _load_chunk_index(cfg.comment_chunks, cfg.post_chunks)

    missing = [cid for cid in positive_ids if cid not in chunk_index]
    if missing:
        raise ValueError(
            f"{len(missing)} positive chunk_ids not found in chunk JSONL files. "
            f"Example: {missing[:3]}"
        )

    pool_by_stratum: dict[str, list[str]] = defaultdict(list)
    for cid, rec in chunk_index.items():
        if cid in positive_set:
            continue
        pool_by_stratum[_stratum_key(rec)].append(cid)

    log.info("Sampling %s distractors (stratified)…", n_neg)
    distractor_ids = _stratified_sample_distractors(
        pool_by_stratum=pool_by_stratum,
        n_need=n_neg,
        rng=rng,
    )

    final_ids = positive_ids + distractor_ids
    rng.shuffle(final_ids)

    cfg.out_corpus.parent.mkdir(parents=True, exist_ok=True)
    with cfg.out_corpus.open("w", encoding="utf-8") as fh:
        for cid in final_ids:
            rec = chunk_index[cid]
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    meta = EvalCorpusMeta(
        target_total=target_total,
        positive_cap=pos_cap,
        positive_fraction=POSITIVE_FRACTION,
        seed=cfg.seed,
        unique_positives_in_corpus=n_pos,
        distractor_count=n_neg,
        total_lines_written=len(final_ids),
        comment_chunks_source=str(cfg.comment_chunks.resolve()),
        post_chunks_source=str(cfg.post_chunks.resolve()),
        suggest_max_per_query=cfg.suggest_max_per_query,
        query_id_to_candidate_positives=query_pos_map,
    )
    cfg.out_meta.write_text(meta.model_dump_json(indent=2), encoding="utf-8")
    positives_payload = {
        "version": 1,
        "description": "Per-query candidate chunk_ids from suggest_eval_chunks ranking (not deduped across queries).",
        "query_id_to_candidate_positives": query_pos_map,
    }
    cfg.out_positives.write_text(
        json.dumps(positives_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info(
        "Wrote %s lines (%s positives, %s distractors) → %s",
        len(final_ids),
        n_pos,
        n_neg,
        cfg.out_corpus,
    )
    log.info("Wrote meta → %s", cfg.out_meta)
    log.info("Wrote positives map → %s", cfg.out_positives)
    return meta


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="Build eval corpus (positives + distractors).")
    ap.add_argument("--golden", type=Path, default=root / "data" / "golden_queries.json")
    ap.add_argument(
        "--comment-chunks",
        type=Path,
        default=None,
        help="Comment chunk JSONL for corpus bodies (default: data/comment_chunks.jsonl — post-enrichment canonical)",
    )
    ap.add_argument("--post-chunks", type=Path, default=root / "data" / "post_chunks.jsonl")
    ap.add_argument(
        "--suggest-comment-chunks",
        type=Path,
        default=None,
        help="Chunk JSONL used for candidate ranking (default: same as --comment-chunks)",
    )
    ap.add_argument(
        "--suggest-post-chunks",
        type=Path,
        default=None,
        help="Post chunk JSONL used for candidate ranking (default: same as --post-chunks)",
    )
    ap.add_argument("--out-corpus", type=Path, default=root / "data" / "eval_corpus.jsonl")
    ap.add_argument("--out-meta", type=Path, default=root / "data" / "eval_corpus_meta.json")
    ap.add_argument(
        "--out-positives",
        type=Path,
        default=root / "data" / "eval_corpus_positives.json",
    )
    ap.add_argument("--target-total", type=int, default=2000)
    ap.add_argument("--positive-cap", type=int, default=500)
    ap.add_argument("--max-per-query", type=int, default=25)
    ap.add_argument("--preview-len", type=int, default=180)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    comment = args.comment_chunks or (root / "data" / "comment_chunks.jsonl")
    suggest_comment = args.suggest_comment_chunks or comment
    post = args.post_chunks
    suggest_post = args.suggest_post_chunks or post

    if not args.golden.exists():
        log.error("Missing golden queries: %s", args.golden)
        sys.exit(1)

    cfg = BuildConfig(
        root=root,
        golden_path=args.golden,
        comment_chunks=comment,
        post_chunks=post,
        suggest_comment_chunks=suggest_comment,
        suggest_post_chunks=suggest_post,
        out_corpus=args.out_corpus,
        out_meta=args.out_meta,
        out_positives=args.out_positives,
        target_total=args.target_total,
        positive_cap=args.positive_cap,
        suggest_max_per_query=args.max_per_query,
        text_preview_len=args.preview_len,
        seed=args.seed,
    )
    try:
        run(cfg)
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
