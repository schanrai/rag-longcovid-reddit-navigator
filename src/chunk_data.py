#!/usr/bin/env python3
"""
chunk_data.py — Long COVID Reddit RAG: Comment chunking pipeline (v1)

Two-pass pipeline:
  Pass 1 — Stream all comments to compute per-parent agreement_count and
            thanks_count from Gate 2 failures. Heuristics reused from
            gate_analysis.py to guarantee parity with analysis counts.
  Pass 2 — Apply content exclusions, Gate 2, and sliding-window chunking.
            Attach post_title context + social signal counts to every chunk.

Reads:
  data/r_LongCovid_comments.jsonl
  data/r_LongCovid_posts.jsonl

Writes:
  data/chunks.jsonl          — NDJSON, one chunk record per line
  reports/chunk_report.json  — pipeline summary (parameters + metrics)

Ingestion rules locked — see:
  docs/gate_analysis_findings.md  Sections 9–12  (exclusions, score floor, signals)
  long-covid-rag-scope-v2.md      Section 4.3    (ingestion flow diagram)
                                  Section 6      (chunking parameters)

Usage:
  python3 src/chunk_data.py
"""
from __future__ import annotations

import json
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Generator

# Reuse validated heuristics and streaming utilities from gate_analysis.py
# so agreement/thanks detection stays in sync with the analysis report numbers.
sys.path.insert(0, str(Path(__file__).parent))
from gate_analysis import (
    Config as GateConfig,
    is_agreement,
    is_thanks,
    stream_ndjson,
    word_count,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("chunk_data")

# Shared GateConfig instance — carries the heuristic phrase lists
_GATE_CFG = GateConfig()


# ── Configuration ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Config:
    data_dir: Path = Path(__file__).parent.parent / "data"
    reports_dir: Path = Path(__file__).parent.parent / "reports"
    comments_file: str = "r_LongCovid_comments.jsonl"
    posts_file: str = "r_LongCovid_posts.jsonl"
    chunks_out: str = "chunks.jsonl"
    report_out: str = "chunk_report.json"

    # Gate 2 — locked (gate_analysis_findings.md D1)
    gate2_words: int = 25

    # Score floor — locked (gate_analysis_findings.md D2, Section 7)
    # score < score_floor → excluded. score=0 (contested) is indexed.
    score_floor: int = 0

    # Chunking — locked (long-covid-rag-scope-v2.md Section 6.1)
    chunk_max_words: int = 200
    chunk_overlap_words: int = 20

    progress_interval: int = 25_000


# ── Content exclusion ──────────────────────────────────────────────────────────

class ExclusionReason(str, Enum):
    BODY_EMPTY = "body_empty"
    BODY_DELETED = "body_deleted"
    BODY_REMOVED = "body_removed"
    DISTINGUISHED_MODERATOR = "distinguished_moderator"
    SCORE_NEGATIVE = "score_negative"


_IRRECOVERABLE_BODIES: frozenset[str] = frozenset({"[deleted]", "[removed]"})


def content_exclusion(record: dict[str, Any], cfg: Config) -> ExclusionReason | None:
    """
    Returns the ExclusionReason for a comment, or None if it passes all checks.

    Order follows the locked ingestion flow (gate_analysis_findings.md Section 12):
      body deleted/removed/empty → distinguished moderator → score < 0 → pass
    """
    body: str = (record.get("body") or "").strip()

    if not body:
        return ExclusionReason.BODY_EMPTY
    if body == "[deleted]":
        return ExclusionReason.BODY_DELETED
    if body == "[removed]":
        return ExclusionReason.BODY_REMOVED

    if record.get("distinguished") == "moderator":
        return ExclusionReason.DISTINGUISHED_MODERATOR

    if int(record.get("score") or 0) < cfg.score_floor:
        return ExclusionReason.SCORE_NEGATIVE

    return None


# ── Chunking ───────────────────────────────────────────────────────────────────

def split_into_chunks(words: list[str], max_words: int, overlap: int) -> list[list[str]]:
    """
    Sliding-window word-level chunker (long-covid-rag-scope-v2.md Section 6.1).

    - len(words) <= max_words  →  single chunk, no splitting.
    - len(words) > max_words   →  stride = max_words - overlap. Chunks produced
      until the entire word list is covered. The last chunk may be shorter than
      max_words; this is acceptable because the parent comment already passed Gate 2.

    Args:
        words:     body.split() — whitespace-tokenised word list.
        max_words: Maximum words per chunk (locked: 200).
        overlap:   Words shared with the preceding chunk (locked: 20).

    Returns:
        Non-empty list of word lists. Each inner list represents one chunk.
    """
    if len(words) <= max_words:
        return [words]

    stride = max_words - overlap
    chunks: list[list[str]] = []
    start = 0
    while start < len(words):
        chunks.append(words[start : start + max_words])
        if start + max_words >= len(words):
            break
        start += stride
    return chunks


def build_chunk_record(
    comment_id: str,
    record: dict[str, Any],
    chunk_words: list[str],
    chunk_index: int,
    total_chunks: int,
    post_title: str,
    agreement_count: int,
    thanks_count: int,
) -> dict[str, Any]:
    """
    Assembles one chunk output record per the schema in Section 6.2.

    Atoms are stored separately (text, post_title, post_summary) so that the
    embedding script can compose them at compute time without re-running chunking.
    post_summary is null in v1 — filled by enrich_summaries.py in a later pass.
    """
    return {
        "chunk_id": f"t1_{comment_id}_{chunk_index}",
        "text": " ".join(chunk_words),
        "post_title": post_title,
        "post_summary": None,
        "comment_id": comment_id,
        "comment_score": int(record.get("score") or 0),
        "agreement_count": agreement_count,
        "thanks_count": thanks_count,
        "parent_id": record.get("parent_id", ""),
        "link_id": record.get("link_id", ""),
        "nest_level": int(record.get("nest_level") or 0),
        "is_submitter": bool(record.get("is_submitter", False)),
        "stickied": bool(record.get("stickied", False)),
        "created_utc": record.get("created_utc"),
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "word_count": len(chunk_words),
    }


# ── Data loading ───────────────────────────────────────────────────────────────

def load_post_titles(posts_path: Path) -> dict[str, str]:
    """
    Load post id → title into memory. Bare post ID (no 't3_' prefix) is the key,
    matching the link_id[3:] join pattern used in Pass 2.

    25,907 posts with id + title ≈ a few MB — fits comfortably in RAM.
    Returns a plain dict (not defaultdict) so missing keys are explicit.
    """
    titles: dict[str, str] = {}
    missing = 0
    for _, record, error in stream_ndjson(posts_path):
        if error or record is None:
            continue
        post_id = record.get("id")
        if not post_id:
            continue
        title = (record.get("title") or "").strip()
        if not title:
            missing += 1
        titles[post_id] = title

    log.info(f"  Posts loaded: {len(titles):,}  (missing titles: {missing})")
    return titles


# ── Pass 1: Social signal computation ─────────────────────────────────────────

def compute_social_signals(
    comments_path: Path,
    cfg: Config,
) -> tuple[dict[str, int], dict[str, int]]:
    """
    Pass 1 — Compute per-parent agreement and thanks counts from Gate 2 failures.

    Only comments that pass content checks AND fail Gate 2 (< gate2_words) are
    evaluated. This follows the ingestion flow order exactly:
      content exclusion → gate2 → social signal heuristics

    Keys in returned dicts are full Reddit IDs (e.g. "t1_abc123", "t3_xyz789").
    A comment chunk's own social signal count is looked up via "t1_{comment_id}".

    Returns:
        (agreement_by_parent, thanks_by_parent) — plain dicts, int counts.
    """
    agreement_by_parent: dict[str, int] = defaultdict(int)
    thanks_by_parent: dict[str, int] = defaultdict(int)
    processed = 0

    log.info("=== Pass 1: Computing social signals ===")

    for _, record, error in stream_ndjson(comments_path):
        if error or record is None:
            continue

        processed += 1
        if processed % cfg.progress_interval == 0:
            log.info(f"  Pass 1 — {processed:,} comments scanned …")

        if content_exclusion(record, cfg) is not None:
            continue

        body: str = (record.get("body") or "").strip()
        wc = word_count(body)

        if wc >= cfg.gate2_words:
            continue  # will be indexed — does not contribute to social signals

        parent_id: str = record.get("parent_id") or ""
        if not parent_id:
            continue

        if is_agreement(body, _GATE_CFG):
            agreement_by_parent[parent_id] += 1

        if is_thanks(body, _GATE_CFG):
            thanks_by_parent[parent_id] += 1

    log.info(
        f"  Pass 1 done — {processed:,} comments scanned  |  "
        f"agreement parents: {len(agreement_by_parent):,}  |  "
        f"thanks parents: {len(thanks_by_parent):,}"
    )
    return dict(agreement_by_parent), dict(thanks_by_parent)


# ── Pass 2: Chunking ───────────────────────────────────────────────────────────

@dataclass
class ChunkStats:
    total_comments: int = 0
    exclusions: Counter = field(default_factory=Counter)
    gate2_failures: int = 0
    comments_indexed: int = 0
    total_chunks: int = 0
    single_chunk_comments: int = 0
    multi_chunk_comments: int = 0
    max_chunks_single_comment: int = 0
    missing_post_title: int = 0


def _iter_chunks(
    comments_path: Path,
    post_titles: dict[str, str],
    agreement_by_parent: dict[str, int],
    thanks_by_parent: dict[str, int],
    cfg: Config,
    stats: ChunkStats,
) -> Generator[dict[str, Any], None, None]:
    """
    Inner generator for Pass 2. Mutates stats in place as it streams.
    Yielded records are written directly to disk by the caller — no in-memory
    accumulation of the full chunk list.
    """
    for _, record, error in stream_ndjson(comments_path):
        if error or record is None:
            continue

        stats.total_comments += 1
        if stats.total_comments % cfg.progress_interval == 0:
            log.info(
                f"  Pass 2 — {stats.total_comments:,} comments  |  "
                f"indexed: {stats.comments_indexed:,}  |  "
                f"chunks: {stats.total_chunks:,}"
            )

        # ── Content exclusion ──────────────────────────────────────────────────
        reason = content_exclusion(record, cfg)
        if reason is not None:
            stats.exclusions[reason.value] += 1
            continue

        body: str = (record.get("body") or "").strip()
        wc = word_count(body)

        # ── Gate 2 ─────────────────────────────────────────────────────────────
        if wc < cfg.gate2_words:
            stats.gate2_failures += 1
            continue

        # ── Context enrichment ─────────────────────────────────────────────────
        link_id: str = record.get("link_id") or ""
        bare_post_id = link_id[3:] if link_id.startswith("t3_") else link_id
        post_title = post_titles.get(bare_post_id, "")
        if not post_title:
            stats.missing_post_title += 1

        comment_id: str = record.get("id") or ""
        comment_full_id = f"t1_{comment_id}"

        # Social signals: how many short replies referenced this specific comment
        agreement_count = agreement_by_parent.get(comment_full_id, 0)
        thanks_count = thanks_by_parent.get(comment_full_id, 0)

        # ── Chunk splitting ────────────────────────────────────────────────────
        words = body.split()
        comment_chunks = split_into_chunks(words, cfg.chunk_max_words, cfg.chunk_overlap_words)
        n_chunks = len(comment_chunks)

        stats.comments_indexed += 1
        stats.total_chunks += n_chunks
        if n_chunks == 1:
            stats.single_chunk_comments += 1
        else:
            stats.multi_chunk_comments += 1
            if n_chunks > stats.max_chunks_single_comment:
                stats.max_chunks_single_comment = n_chunks

        for idx, chunk_words in enumerate(comment_chunks):
            yield build_chunk_record(
                comment_id=comment_id,
                record=record,
                chunk_words=chunk_words,
                chunk_index=idx,
                total_chunks=n_chunks,
                post_title=post_title,
                agreement_count=agreement_count,
                thanks_count=thanks_count,
            )

    log.info(
        f"  Pass 2 done — {stats.total_comments:,} comments  |  "
        f"{stats.comments_indexed:,} indexed  |  {stats.total_chunks:,} chunks"
    )


def run_chunking_pass(
    comments_path: Path,
    post_titles: dict[str, str],
    agreement_by_parent: dict[str, int],
    thanks_by_parent: dict[str, int],
    cfg: Config,
    out_path: Path,
) -> ChunkStats:
    """
    Pass 2 entry point. Streams chunk records directly to disk to avoid
    accumulating ~160k chunk dicts in memory.

    Returns the populated ChunkStats after all comments are processed.
    """
    stats = ChunkStats()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fh:
        for chunk_record in _iter_chunks(
            comments_path, post_titles, agreement_by_parent, thanks_by_parent, cfg, stats
        ):
            fh.write(json.dumps(chunk_record, ensure_ascii=False) + "\n")

    return stats


# ── Report ─────────────────────────────────────────────────────────────────────

def _pct(count: int, total: int) -> float:
    return round(count / total * 100, 4) if total else 0.0


def build_report(
    stats: ChunkStats,
    agreement_by_parent: dict[str, int],
    thanks_by_parent: dict[str, int],
    cfg: Config,
) -> dict[str, Any]:
    total = stats.total_comments
    total_excluded = sum(stats.exclusions.values())
    avg_chunks = (
        round(stats.total_chunks / stats.comments_indexed, 3)
        if stats.comments_indexed
        else 0.0
    )

    return {
        "schema": "chunk_report_v1",
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "parameters": {
            "gate2_threshold_words": cfg.gate2_words,
            "score_floor": f"comment_score >= {cfg.score_floor}",
            "chunk_max_words": cfg.chunk_max_words,
            "chunk_overlap_words": cfg.chunk_overlap_words,
            "note": (
                "Gate 1 (post score ratio) dropped — comment.score stored as "
                "metadata for ranking. Only score < 0 excluded at ingestion."
            ),
        },
        "input": {
            "total_comments": total,
        },
        "exclusions": {
            reason: {
                "count": count,
                "pct_of_total": _pct(count, total),
            }
            for reason, count in stats.exclusions.items()
        }
        | {
            "total": {
                "count": total_excluded,
                "pct_of_total": _pct(total_excluded, total),
            }
        },
        "gate2_failures": {
            "count": stats.gate2_failures,
            "pct_of_total": _pct(stats.gate2_failures, total),
            "note": "word count < 25. Social signals extracted before discarding.",
        },
        "social_signals_pass1": {
            "agreement_by_parent_unique": len(agreement_by_parent),
            "thanks_by_parent_unique": len(thanks_by_parent),
            "note": (
                "Counts computed from Gate 2 failures only, after content exclusions. "
                "Attached to indexed comment chunks via t1_{comment_id} lookup."
            ),
        },
        "indexed": {
            "comments_indexed": stats.comments_indexed,
            "pct_of_total": _pct(stats.comments_indexed, total),
            "total_chunks": stats.total_chunks,
            "single_chunk_comments": stats.single_chunk_comments,
            "multi_chunk_comments": stats.multi_chunk_comments,
            "max_chunks_single_comment": stats.max_chunks_single_comment,
            "avg_chunks_per_comment": avg_chunks,
            "missing_post_title": stats.missing_post_title,
        },
    }


def print_summary(report: dict[str, Any]) -> None:
    p = report["parameters"]
    idx = report["indexed"]
    excl = report["exclusions"]
    sig = report["social_signals_pass1"]

    div = "═" * 72
    print(f"\n{div}")
    print("  CHUNK PIPELINE REPORT (v1)")
    print(div)
    print(
        f"  Gate 2: ≥ {p['gate2_threshold_words']} words  |  "
        f"Chunk: ≤ {p['chunk_max_words']} words, {p['chunk_overlap_words']}-word overlap"
    )
    print(div)

    total = report["input"]["total_comments"]
    print(f"\n  {'Metric':<46} {'Count':>12} {'%':>12}")
    print(f"  {'─' * 46} {'─' * 12} {'─' * 12}")
    print(f"  {'Total comments':<46} {total:>12,} {'100.0000%':>12}")

    for key in (
        ExclusionReason.BODY_DELETED.value,
        ExclusionReason.BODY_REMOVED.value,
        ExclusionReason.BODY_EMPTY.value,
        ExclusionReason.DISTINGUISHED_MODERATOR.value,
        ExclusionReason.SCORE_NEGATIVE.value,
    ):
        if key in excl:
            e = excl[key]
            print(f"  {'  Excluded: ' + key:<46} {e['count']:>12,} {e['pct_of_total']:>11.4f}%")

    gate2 = report["gate2_failures"]
    print(f"  {'Gate 2 failures (< 25 words)':<46} {gate2['count']:>12,} {gate2['pct_of_total']:>11.4f}%")
    print(
        f"  {'  ↳ agreement parents detected':<46} {sig['agreement_by_parent_unique']:>12,}"
    )
    print(
        f"  {'  ↳ thanks parents detected':<46} {sig['thanks_by_parent_unique']:>12,}"
    )
    print(f"  {'Comments indexed':<46} {idx['comments_indexed']:>12,} {idx['pct_of_total']:>11.4f}%")
    print(f"  {'Total chunks produced':<46} {idx['total_chunks']:>12,}")
    print(f"  {'  Single-chunk comments':<46} {idx['single_chunk_comments']:>12,}")
    print(f"  {'  Multi-chunk comments':<46} {idx['multi_chunk_comments']:>12,}")
    print(f"  {'  Max chunks (one comment)':<46} {idx['max_chunks_single_comment']:>12,}")
    print(f"  {'  Avg chunks per comment':<46} {idx['avg_chunks_per_comment']:>12.3f}")
    print(f"  {'  Missing post title':<46} {idx['missing_post_title']:>12,}")

    print(f"\n{div}\n")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = Config()
    comments_path = cfg.data_dir / cfg.comments_file
    posts_path = cfg.data_dir / cfg.posts_file

    for path in (comments_path, posts_path):
        if not path.exists():
            log.error(f"File not found: {path}")
            sys.exit(1)

    cfg.reports_dir.mkdir(parents=True, exist_ok=True)

    # ── Load posts ─────────────────────────────────────────────────────────────
    log.info(f"=== Loading post titles: {posts_path.name} ===")
    post_titles = load_post_titles(posts_path)

    # ── Pass 1: Social signals ─────────────────────────────────────────────────
    agreement_by_parent, thanks_by_parent = compute_social_signals(comments_path, cfg)

    # ── Pass 2: Chunking ───────────────────────────────────────────────────────
    chunks_path = cfg.data_dir / cfg.chunks_out
    log.info(f"=== Pass 2: Chunking → {chunks_path} ===")
    stats = run_chunking_pass(
        comments_path=comments_path,
        post_titles=post_titles,
        agreement_by_parent=agreement_by_parent,
        thanks_by_parent=thanks_by_parent,
        cfg=cfg,
        out_path=chunks_path,
    )

    # ── Report ─────────────────────────────────────────────────────────────────
    report = build_report(stats, agreement_by_parent, thanks_by_parent, cfg)
    report_path = cfg.reports_dir / cfg.report_out
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print_summary(report)
    log.info(f"Chunk report → {report_path}")
    log.info(f"Chunks NDJSON → {chunks_path}")
    print(
        f"✅  {stats.comments_indexed:,} comments indexed  →  "
        f"{stats.total_chunks:,} chunks written to {chunks_path}\n"
    )


if __name__ == "__main__":
    main()
