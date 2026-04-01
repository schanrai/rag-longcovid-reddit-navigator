#!/usr/bin/env python3
"""
gate_analysis.py — Long COVID Reddit RAG: Gate analysis (v3)

Locked parameters (see gate_analysis_findings.md v4):
  Gate 2: comment word count ≥ 25 (substance)
  Score floor: comment score ≥ 0 (binary; negative scores excluded from indexable corpus)
  Former Gate 1 (post score ratio) dropped — comment.score stored as metadata, used as ranking signal.

New in v3 (gate_analysis_findings.md Section 8 & 10):
  - thanks_count heuristic added (utility signal — parallel to agreement heuristic)
  - agreement_count and thanks_count now tracked per parent_id (t3_ post or t1_ comment)
    Architectural correction: signals can reply to comments (t1_) not just posts (t3_)
  - Per-parent dicts serialised to report for use by chunk_data.py at ingestion

Outputs:
  gate_analysis_report_v3.json

Usage:
  python3 gate_analysis.py
"""
from __future__ import annotations

import json
import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("gate_analysis")


# ── Configuration ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Config:
    data_dir: Path = Path(__file__).parent.parent / "data"
    reports_dir: Path = Path(__file__).parent.parent / "reports"
    comments_file: str = "r_LongCovid_comments.jsonl"

    # Locked Gate 2 threshold (words)
    gate2_words: int = 25

    # Collect word-count boundary samples (gate2_words ± margin)
    boundary_margin: int = 5
    boundary_sample_size: int = 10

    # ── Agreement heuristic (prevalence signal) ────────────────────────────────
    # Keyword-based detection on comments failing Gate 2 (< gate2_words)
    agreement_phrases: tuple[str, ...] = (
        "same here", "me too", "+1", "same for me", "can relate",
        "yes same", "yeah same", "yea same", "omg same", "omg yes",
        "so much this", "this exactly", "exactly this",
        "same 🙋", "same 😔", "same 😭", "same 😢",
        "me 🙋", "me 🙋‍♀️", "me 🙋‍♂️", "me 😔", "me 😭", "me 😢",
        "yes this", "yeah this", "yep this",
        "same tbh", "same lol", "same honestly",
        "following", "subscribed",
    )
    agreement_exact: frozenset[str] = frozenset({
        "same", "this", "agreed", "exactly", "yep", "yup",
        "absolutely", "definitely", "yes", "yeah", "yea",
        "same.", "yes.", "yeah.", "same!", "this!", "yes!!", "same!!",
        "indeed", "yessss", "same...", "true", "true.", "totally",
        "me",
    })

    # ── Thanks heuristic (utility signal) ─────────────────────────────────────
    # Expresses gratitude/acknowledgement — indicates parent content was actionable/helpful
    thanks_phrases: tuple[str, ...] = (
        "thank you", "thanks for", "really appreciate", "this helped",
        "so helpful", "very helpful", "this is helpful", "that's helpful",
        "thats helpful", "was helpful", "were helpful", "helped me",
        "helped a lot", "appreciate you", "appreciate the", "appreciate it",
        "appreciate that", "appreciate this", "thank u", "thanks so much",
        "thank you so much", "thanks a lot", "thanks a bunch",
        "grateful for", "grateful to", "so grateful",
    )
    thanks_exact: frozenset[str] = frozenset({
        "ty", "tysm", "tyvm", "thank you", "thanks", "thx", "thnx",
        "thank you!", "thanks!", "ty!", "tysm!", "thx!",
        "thank you so much", "thanks so much",
    })

    # Word-count buckets for Gate 2 failures only (< gate2_words)
    short_buckets: tuple[tuple[int, int, str], ...] = (
        (0, 5, "0-5"),
        (6, 15, "6-15"),
        (16, 24, "16-24"),
    )

    bucket_sample_size: int = 5
    progress_interval: int = 25_000
    gate_report_out: str = "gate_analysis_report_v3.json"


# ── NDJSON reader ──────────────────────────────────────────────────────────────

def stream_ndjson(
    path: Path,
) -> Generator[tuple[int, dict[str, Any] | None, str | None], None, None]:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line_num, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield line_num, json.loads(raw), None
            except json.JSONDecodeError as exc:
                yield line_num, None, f"line {line_num}: {exc}"


def word_count(text: str) -> int:
    return len(text.split()) if text else 0


_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def is_agreement(body: str, config: Config) -> bool:
    """
    Returns True if a short comment expresses agreement/prevalence ("same here", "me too").
    Used for agreement_count metadata on the parent document at ingestion.
    """
    normalized = (body or "").strip().lower()
    if not normalized:
        return True

    for phrase in config.agreement_phrases:
        if phrase in normalized:
            return True

    clean = _PUNCT_RE.sub("", normalized).strip()
    if not clean:
        return True

    if clean in config.agreement_exact:
        return True

    words = clean.split()
    _affirmation_openers = {
        "yes", "yeah", "yea", "yep", "yup", "same", "true",
        "totally", "absolutely", "definitely", "agreed", "exactly",
    }
    if len(words) <= 5 and words[0] in _affirmation_openers:
        return True

    return False


def is_thanks(body: str, config: Config) -> bool:
    """
    Returns True if a short comment expresses gratitude/utility ("thank you", "this helped").
    Distinct from agreement: agreement = prevalence signal, thanks = utility signal.
    Used for thanks_count metadata on the parent document at ingestion.
    """
    normalized = (body or "").strip().lower()
    if not normalized:
        return False  # empty comments are agreement, not thanks

    for phrase in config.thanks_phrases:
        if phrase in normalized:
            return True

    clean = _PUNCT_RE.sub("", normalized).strip()
    if not clean:
        return False

    if clean in config.thanks_exact:
        return True

    words = clean.split()
    _thanks_openers = {"thank", "thanks", "grateful", "appreciate"}
    if len(words) <= 5 and words[0] in _thanks_openers:
        return True

    return False


def _bucket_label(wc: int, config: Config) -> str:
    for lo, hi, label in config.short_buckets:
        if lo <= wc <= hi:
            return label
    return "other"


def _make_short_buckets(config: Config) -> dict[str, dict[str, Any]]:
    return {
        label: {
            "total": 0,
            "agreement": 0,
            "thanks": 0,
            "neither": 0,
            "samples": [],
        }
        for _, _, label in config.short_buckets
    }


def _parent_type(parent_id: str) -> str:
    """Return 't3' (post), 't1' (comment), or 'unknown'."""
    if parent_id.startswith("t3_"):
        return "t3"
    if parent_id.startswith("t1_"):
        return "t1"
    return "unknown"


# ── Analysis accumulator ────────────────────────────────────────────────────────

@dataclass
class AnalysisResult:
    total_comments: int = 0
    word_counts: list[int] = field(default_factory=list)
    comment_scores: list[int] = field(default_factory=list)

    boundary_samples: list[dict[str, Any]] = field(default_factory=list)

    # Agreement (prevalence signal)
    agreement_count: int = 0
    agreement_to_post: int = 0    # parent_id starts with t3_
    agreement_to_comment: int = 0 # parent_id starts with t1_
    agreement_by_parent: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    agreement_samples: list[dict[str, Any]] = field(default_factory=list)

    # Thanks (utility signal)
    thanks_count: int = 0
    thanks_to_post: int = 0
    thanks_to_comment: int = 0
    thanks_by_parent: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    thanks_samples: list[dict[str, Any]] = field(default_factory=list)

    # Short comments that are neither agreement nor thanks (noise)
    neither_short_count: int = 0
    neither_samples: list[dict[str, Any]] = field(default_factory=list)

    short_by_bucket: dict[str, dict[str, Any]] = field(default_factory=dict)


def run_analysis(config: Config) -> AnalysisResult:
    path = config.data_dir / config.comments_file
    result = AnalysisResult()
    g2 = config.gate2_words
    boundary_low = g2 - config.boundary_margin
    boundary_high = g2 + config.boundary_margin

    result.short_by_bucket = _make_short_buckets(config)

    log.info(
        f"=== Streaming comments: {path.name} "
        f"(Gate 2 ≥ {g2} wds | score floor ≥ 0 | agreement + thanks signals) ==="
    )

    for _, record, error in stream_ndjson(path):
        if error or record is None:
            continue

        result.total_comments += 1

        body: str = record.get("body") or ""
        comment_score = int(record.get("score") or 0)
        parent_id: str = record.get("parent_id") or ""

        wc = word_count(body)
        result.word_counts.append(wc)
        result.comment_scores.append(comment_score)

        if boundary_low <= wc <= boundary_high and len(result.boundary_samples) < config.boundary_sample_size:
            result.boundary_samples.append({
                "id": record.get("id"),
                "word_count": wc,
                "comment_score": comment_score,
                "parent_id": parent_id,
                "body": body[:300],
            })

        if wc < g2:
            bucket = _bucket_label(wc, config)
            if bucket == "other":
                continue
            bkt = result.short_by_bucket[bucket]
            bkt["total"] += 1

            agreed = is_agreement(body, config)
            thanked = is_thanks(body, config)
            ptype = _parent_type(parent_id)

            if agreed:
                result.agreement_count += 1
                bkt["agreement"] += 1
                result.agreement_by_parent[parent_id] += 1
                if ptype == "t3":
                    result.agreement_to_post += 1
                elif ptype == "t1":
                    result.agreement_to_comment += 1
                if len(result.agreement_samples) < 10:
                    result.agreement_samples.append({
                        "id": record.get("id"),
                        "word_count": wc,
                        "parent_id": parent_id,
                        "parent_type": ptype,
                        "body": body[:200],
                    })

            if thanked:
                result.thanks_count += 1
                bkt["thanks"] += 1
                result.thanks_by_parent[parent_id] += 1
                if ptype == "t3":
                    result.thanks_to_post += 1
                elif ptype == "t1":
                    result.thanks_to_comment += 1
                if len(result.thanks_samples) < 10:
                    result.thanks_samples.append({
                        "id": record.get("id"),
                        "word_count": wc,
                        "parent_id": parent_id,
                        "parent_type": ptype,
                        "body": body[:200],
                    })

            if not agreed and not thanked:
                result.neither_short_count += 1
                bkt["neither"] += 1
                if len(result.neither_samples) < 10:
                    result.neither_samples.append({
                        "id": record.get("id"),
                        "word_count": wc,
                        "body": body[:200],
                    })
                if len(bkt["samples"]) < config.bucket_sample_size:
                    bkt["samples"].append({
                        "id": record.get("id"),
                        "word_count": wc,
                        "body": body[:200],
                    })

        if result.total_comments % config.progress_interval == 0:
            log.info(f"  Comments: {result.total_comments:,} processed …")

    log.info(f"  Done — {result.total_comments:,} comments analyzed")
    return result


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(result: AnalysisResult, config: Config) -> dict[str, Any]:
    total = result.total_comments
    g2 = config.gate2_words
    wcs = result.word_counts
    scores = result.comment_scores

    pass_g2 = sum(1 for wc in wcs if wc >= g2)
    pass_floor = sum(1 for s in scores if s >= 0)
    excluded_neg = sum(1 for s in scores if s < 0)
    pass_both = sum(1 for wc, s in zip(wcs, scores) if wc >= g2 and s >= 0)

    def pct(count: int) -> float:
        return round(count / total * 100, 4) if total else 0.0

    gate2_failures = total - pass_g2

    def pct_of(count: int, denom: int) -> float:
        return round(count / denom * 100, 4) if denom else 0.0

    buckets_out: dict[str, Any] = {}
    for label, bkt in result.short_by_bucket.items():
        t = bkt["total"]
        buckets_out[label] = {
            "total": t,
            "agreement": bkt["agreement"],
            "agreement_pct": pct_of(bkt["agreement"], t),
            "thanks": bkt["thanks"],
            "thanks_pct": pct_of(bkt["thanks"], t),
            "neither": bkt["neither"],
            "neither_pct": pct_of(bkt["neither"], t),
        }

    return {
        "total_comments": total,
        "pass_gate2": {
            "threshold_words": g2,
            "count": pass_g2,
            "pct_of_total": pct(pass_g2),
        },
        "pass_score_floor": {
            "rule": "comment_score >= 0",
            "count": pass_floor,
            "pct_of_total": pct(pass_floor),
        },
        "pass_both_indexable_corpus": {
            "count": pass_both,
            "pct_of_total": pct(pass_both),
        },
        "excluded_score_negative": {
            "count": excluded_neg,
            "pct_of_total": pct(excluded_neg),
        },
        "agreement_signal": {
            "description": "Prevalence signal — short comments expressing 'same here', 'me too'",
            "gate2_failure_count": gate2_failures,
            "total": result.agreement_count,
            "pct_of_gate2_failures": pct_of(result.agreement_count, gate2_failures),
            "to_post_t3": result.agreement_to_post,
            "to_comment_t1": result.agreement_to_comment,
            "pct_to_comment": pct_of(result.agreement_to_comment, result.agreement_count),
            "unique_parents": len(result.agreement_by_parent),
        },
        "thanks_signal": {
            "description": "Utility signal — short comments expressing gratitude/acknowledgement",
            "gate2_failure_count": gate2_failures,
            "total": result.thanks_count,
            "pct_of_gate2_failures": pct_of(result.thanks_count, gate2_failures),
            "to_post_t3": result.thanks_to_post,
            "to_comment_t1": result.thanks_to_comment,
            "pct_to_comment": pct_of(result.thanks_to_comment, result.thanks_count),
            "unique_parents": len(result.thanks_by_parent),
        },
        "word_count_buckets_gate2_failures": buckets_out,
    }


def pct_stats(data: list[float | int]) -> dict[str, Any]:
    if not data:
        return {"count": 0}
    s = sorted(data)
    n = len(s)

    def q(frac: float) -> float:
        idx = frac * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        return round(s[lo] + (s[hi] - s[lo]) * (idx - lo), 4)

    return {
        "count": n,
        "min": s[0],
        "p25": q(0.25),
        "median": q(0.50),
        "p75": q(0.75),
        "p90": q(0.90),
        "p95": q(0.95),
        "max": s[-1],
    }


# ── Console report ─────────────────────────────────────────────────────────────

def print_report(result: AnalysisResult, metrics: dict[str, Any], config: Config) -> None:
    total = metrics["total_comments"]
    m = metrics
    idx = m["pass_both_indexable_corpus"]["count"]
    idx_pct = m["pass_both_indexable_corpus"]["pct_of_total"]
    ag = m["agreement_signal"]
    th = m["thanks_signal"]
    g2f = ag["gate2_failure_count"]

    div = "═" * 72
    print(f"\n{div}")
    print("  GATE ANALYSIS (v3) — LOCKED THRESHOLDS + SOCIAL SIGNALS")
    print(div)
    print(f"  Gate 2: ≥ {config.gate2_words} words  |  Score floor: comment_score ≥ 0")
    print(div)

    print(f"\n  {'Metric':<46} {'Count':>12} {'%':>12}")
    print(f"  {'─' * 46} {'─' * 12} {'─' * 12}")

    g2 = m["pass_gate2"]
    sf = m["pass_score_floor"]
    both = m["pass_both_indexable_corpus"]
    ex = m["excluded_score_negative"]

    print(f"  {'Total comments':<46} {total:>12,} {'100.0000%':>12}")
    g2_label = f"Pass Gate 2 (≥ {g2['threshold_words']} words)"
    print(f"  {g2_label:<46} {g2['count']:>12,} {g2['pct_of_total']:>11.4f}%")
    print(f"  {'Pass score ≥ 0 floor':<46} {sf['count']:>12,} {sf['pct_of_total']:>11.4f}%")
    print(f"  {'Pass BOTH (indexable corpus)':<46} {both['count']:>12,} {both['pct_of_total']:>11.4f}%")
    print(f"  {'Excluded (score < 0)':<46} {ex['count']:>12,} {ex['pct_of_total']:>11.4f}%")

    print(f"\n  Social signals (Gate 2 failures only, < {config.gate2_words} words = {g2f:,} comments)")
    print(f"  {'─' * 46} {'─' * 12} {'─' * 12}")
    print(f"  {'  (% column = share of Gate 2 failures)':<46}")
    print(f"  {'Agreement (prevalence signal)':<46} "
          f"{ag['total']:>12,} {ag['pct_of_gate2_failures']:>11.4f}%")
    print(f"    ├─ → post (t3_): {ag['to_post_t3']:,}  "
          f"→ comment (t1_): {ag['to_comment_t1']:,}  "
          f"({ag['pct_to_comment']:.1f}% to comments)")
    print(f"    └─ unique parents: {ag['unique_parents']:,}")
    print(f"  {'Thanks (utility signal)':<46} "
          f"{th['total']:>12,} {th['pct_of_gate2_failures']:>11.4f}%")
    print(f"    ├─ → post (t3_): {th['to_post_t3']:,}  "
          f"→ comment (t1_): {th['to_comment_t1']:,}  "
          f"({th['pct_to_comment']:.1f}% to comments)")
    print(f"    └─ unique parents: {th['unique_parents']:,}")

    print(f"\n  Indexable corpus: {idx:,} comments ({idx_pct:.4f}% of total)")

    print(f"\n  Word-count buckets (Gate 2 failures, < {config.gate2_words} words)")
    print(f"  {'Bucket':<10} {'Total':>10} {'Agree':>10} {'Agree %':>9} "
          f"{'Thanks':>10} {'Thanks %':>9} {'Neither':>10}")
    print(f"  {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 9} {'─' * 10} {'─' * 9} {'─' * 10}")
    for label in ("0-5", "6-15", "16-24"):
        b = m["word_count_buckets_gate2_failures"][label]
        print(f"  {label:<10} {b['total']:>10,} {b['agreement']:>10,} "
              f"{b['agreement_pct']:>8.2f}% {b['thanks']:>10,} "
              f"{b['thanks_pct']:>8.2f}% {b['neither']:>10,}")

    print(f"\n  Agreement samples (→ agreement_count metadata on parent doc):")
    for s in result.agreement_samples[:5]:
        ptype = s.get("parent_type", "?")
        print(f"    [{s['word_count']} wds | parent={ptype}] {s['body']!r}")

    print(f"\n  Thanks samples (→ thanks_count metadata on parent doc):")
    for s in result.thanks_samples[:5]:
        ptype = s.get("parent_type", "?")
        print(f"    [{s['word_count']} wds | parent={ptype}] {s['body']!r}")

    print(f"\n  Boundary samples (word count {config.gate2_words}±{config.boundary_margin}):")
    for s in result.boundary_samples[:5]:
        print(f"    [{s['word_count']} wds, score={s['comment_score']}] {s['body'][:100]!r}…")

    print(f"\n{div}\n")


# ── JSON report ────────────────────────────────────────────────────────────────

def build_report(
    result: AnalysisResult,
    metrics: dict[str, Any],
    config: Config,
) -> dict[str, Any]:
    short_total = result.agreement_count + result.thanks_count + result.neither_short_count

    # Serialize per-parent dicts. These are consumed by chunk_data.py to attach
    # agreement_count / thanks_count as metadata fields on indexed chunks.
    # Only parents with count >= 2 are included to keep file size reasonable;
    # single-signal parents are captured in the totals above.
    agreement_by_parent_filtered = {
        pid: cnt for pid, cnt in result.agreement_by_parent.items() if cnt >= 2
    }
    thanks_by_parent_filtered = {
        pid: cnt for pid, cnt in result.thanks_by_parent.items() if cnt >= 2
    }

    return {
        "schema": "gate_analysis_report_v3",
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "parameters": {
            "gate2_threshold_words": config.gate2_words,
            "score_floor": "comment_score >= 0",
            "note": "Former Gate 1 (post score ratio) dropped. comment.score stored as metadata, used as ranking signal at retrieval.",
        },
        "metrics": metrics,
        "word_count_distribution_all_comments": pct_stats(result.word_counts),
        "boundary_samples": result.boundary_samples,
        "agreement_detection": {
            "short_comments_failing_gate2": short_total,
            "total": result.agreement_count,
            "to_post_t3": result.agreement_to_post,
            "to_comment_t1": result.agreement_to_comment,
            "unique_parents": len(result.agreement_by_parent),
            "unique_parents_with_2plus": len(agreement_by_parent_filtered),
            "breakdown_by_word_count_bucket": {
                label: {
                    "total": bkt["total"],
                    "agreement": bkt["agreement"],
                    "agreement_pct": round(
                        bkt["agreement"] / max(bkt["total"], 1) * 100, 4
                    ),
                    "neither_samples": bkt["samples"],
                }
                for label, bkt in result.short_by_bucket.items()
            },
            "agreement_samples": result.agreement_samples,
            "agreement_by_parent_count_gte2": agreement_by_parent_filtered,
        },
        "thanks_detection": {
            "total": result.thanks_count,
            "to_post_t3": result.thanks_to_post,
            "to_comment_t1": result.thanks_to_comment,
            "unique_parents": len(result.thanks_by_parent),
            "unique_parents_with_2plus": len(thanks_by_parent_filtered),
            "breakdown_by_word_count_bucket": {
                label: {
                    "total": bkt["total"],
                    "thanks": bkt["thanks"],
                    "thanks_pct": round(
                        bkt["thanks"] / max(bkt["total"], 1) * 100, 4
                    ),
                }
                for label, bkt in result.short_by_bucket.items()
            },
            "thanks_samples": result.thanks_samples,
            "thanks_by_parent_count_gte2": thanks_by_parent_filtered,
        },
        "neither_short": {
            "count": result.neither_short_count,
            "samples": result.neither_samples,
        },
    }


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    config = Config()
    comments_path = config.data_dir / config.comments_file
    if not comments_path.exists():
        log.error(f"File not found: {comments_path}")
        sys.exit(1)

    result = run_analysis(config)
    metrics = compute_metrics(result, config)

    print_report(result, metrics, config)

    report = build_report(result, metrics, config)
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = config.reports_dir / config.gate_report_out
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log.info(f"Gate analysis report → {out_path}")

    idx = metrics["pass_both_indexable_corpus"]["count"]
    pct = metrics["pass_both_indexable_corpus"]["pct_of_total"]
    print(f"✅  Indexable corpus: {idx:,} comments ({pct:.4f}% of total)\n")


if __name__ == "__main__":
    main()
