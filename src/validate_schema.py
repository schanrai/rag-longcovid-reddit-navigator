#!/usr/bin/env python3
"""
validate_schema.py — Long COVID Reddit RAG: Schema Validation + Date Coverage

Tasks 1 & 2 from data-validation-plan.md. Streams NDJSON files line-by-line
(no full load into memory — comments file is ~524MB).

Outputs:
  schema_report.json   — field presence, thread structure, discovered fields
  coverage_report.json — monthly post/comment counts, gap flags

Usage:
  python3 validate_schema.py
"""
from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator


# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("validate_schema")


# ── Configuration ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Config:
    data_dir: Path = Path(__file__).parent.parent / "data"
    reports_dir: Path = Path(__file__).parent.parent / "reports"

    posts_file: str = "r_LongCovid_posts.jsonl"
    comments_file: str = "r_LongCovid_comments.jsonl"

    required_post_fields: tuple[str, ...] = (
        "id", "title", "selftext", "score",
        "created_utc", "num_comments", "author", "permalink",
    )
    required_comment_fields: tuple[str, ...] = (
        "id", "body", "score", "created_utc",
        "parent_id", "link_id", "author",
    )

    # How many sample records to print / store per file
    sample_count: int = 5
    # How many thread-depth samples to collect per type (direct / nested)
    thread_sample_size: int = 5
    # Log a progress line every N records
    progress_interval: int = 25_000
    # Flag a month whose count drops >50% vs the prior month
    coverage_drop_threshold: float = 0.50

    schema_report_out: str = "schema_report.json"
    coverage_report_out: str = "coverage_report.json"


# ── NDJSON stream reader ────────────────────────────────────────────────────────

def stream_ndjson(
    path: Path,
) -> Generator[tuple[int, dict[str, Any] | None, str | None], None, None]:
    """
    Yield (line_num, record, error) for each non-empty line.
    record is None on JSON parse failure; error is None on success.
    Uses errors='replace' to survive encoding oddities in user-generated text.
    """
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line_num, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield line_num, json.loads(raw), None
            except json.JSONDecodeError as exc:
                yield line_num, None, f"line {line_num}: {exc}"


# ── Field-level helpers ─────────────────────────────────────────────────────────

def field_has_value(record: dict[str, Any], fname: str) -> bool:
    """True if the field exists and is not None, empty string, or whitespace."""
    val = record.get(fname)
    if val is None:
        return False
    if isinstance(val, str) and not val.strip():
        return False
    return True


def field_is_deleted(record: dict[str, Any], fname: str) -> bool:
    """True if the field contains a Reddit deletion sentinel."""
    return record.get(fname) in ("[deleted]", "[removed]")


# ── Schema statistics accumulator ──────────────────────────────────────────────

@dataclass
class SchemaStats:
    record_type: str                                         # "posts" | "comments"
    total: int = 0
    malformed_count: int = 0
    malformed_samples: list[str] = field(default_factory=list)
    field_present: dict[str, int] = field(default_factory=dict)   # count with real value
    field_deleted: dict[str, int] = field(default_factory=dict)   # count of [deleted]/[removed]
    all_field_names: set[str] = field(default_factory=set)         # every key seen
    sample_records: list[dict[str, Any]] = field(default_factory=list)


def _accumulate_schema(
    record: dict[str, Any],
    required_fields: tuple[str, ...],
    stats: SchemaStats,
    config: Config,
) -> None:
    stats.total += 1
    stats.all_field_names.update(record.keys())

    for fname in required_fields:
        if field_has_value(record, fname):
            stats.field_present[fname] = stats.field_present.get(fname, 0) + 1
        elif field_is_deleted(record, fname):
            stats.field_deleted[fname] = stats.field_deleted.get(fname, 0) + 1

    if len(stats.sample_records) < config.sample_count:
        # Store only required fields to keep samples readable
        stats.sample_records.append(
            {k: record.get(k) for k in required_fields}
        )


# ── Thread structure accumulator (comments only) ────────────────────────────────

@dataclass
class ThreadStats:
    direct_replies: int = 0          # parent_id = t3_* (direct reply to a post)
    nested_replies: int = 0          # parent_id = t1_* (reply to another comment)
    unknown_parent: int = 0          # any other prefix, or missing
    link_id_t3_count: int = 0        # link_id = t3_* (should be all of them)
    link_id_other_count: int = 0     # link_id is something else
    direct_samples: list[dict[str, Any]] = field(default_factory=list)
    nested_samples: list[dict[str, Any]] = field(default_factory=list)


def _accumulate_thread(
    record: dict[str, Any],
    thread_stats: ThreadStats,
    config: Config,
) -> None:
    parent_id: str = record.get("parent_id") or ""
    link_id: str = record.get("link_id") or ""

    sample = {
        "id": record.get("id"),
        "parent_id": parent_id,
        "link_id": link_id,
        "body_preview": (record.get("body") or "")[:120],
    }

    if parent_id.startswith("t3_"):
        thread_stats.direct_replies += 1
        if len(thread_stats.direct_samples) < config.thread_sample_size:
            thread_stats.direct_samples.append(sample)
    elif parent_id.startswith("t1_"):
        thread_stats.nested_replies += 1
        if len(thread_stats.nested_samples) < config.thread_sample_size:
            thread_stats.nested_samples.append(sample)
    else:
        thread_stats.unknown_parent += 1

    if link_id.startswith("t3_"):
        thread_stats.link_id_t3_count += 1
    else:
        thread_stats.link_id_other_count += 1


# ── Date coverage helpers ───────────────────────────────────────────────────────

def _to_year_month(created_utc: Any) -> str | None:
    """Convert a Unix timestamp to 'YYYY-MM', return None if unparseable."""
    try:
        dt = datetime.fromtimestamp(int(created_utc), tz=timezone.utc)
        return dt.strftime("%Y-%m")
    except (TypeError, ValueError, OSError):
        return None


def detect_gaps(monthly: dict[str, int], threshold: float) -> list[dict[str, Any]]:
    """
    Return a list of flagged months:
      - count == 0 → flagged as 'zero_records'
      - count drops > threshold vs the prior calendar month
    Note: only flags gaps between consecutive months that are both present in
    the dataset. Missing months in the sequence are also flagged.
    """
    if not monthly:
        return []

    flags: list[dict[str, Any]] = []
    sorted_months = sorted(monthly.keys())

    for i, month in enumerate(sorted_months):
        count = monthly[month]
        if count == 0:
            flags.append({"month": month, "count": 0, "reason": "zero_records"})
        elif i > 0:
            prior_month = sorted_months[i - 1]
            prior_count = monthly.get(prior_month, 0)
            if prior_count > 0:
                drop = (prior_count - count) / prior_count
                if drop > threshold:
                    flags.append({
                        "month": month,
                        "count": count,
                        "prior_month": prior_month,
                        "prior_count": prior_count,
                        "drop_pct": round(drop * 100, 1),
                        "reason": "large_drop",
                    })

    return flags


# ── Validation runners ──────────────────────────────────────────────────────────

def validate_posts(config: Config) -> tuple[SchemaStats, dict[str, int]]:
    path = config.data_dir / config.posts_file
    stats = SchemaStats(record_type="posts")
    monthly: dict[str, int] = defaultdict(int)

    log.info(f"=== Scanning posts: {path.name} ({path.stat().st_size / 1_048_576:.0f} MB) ===")

    for _line_num, record, error in stream_ndjson(path):
        if error:
            stats.malformed_count += 1
            if len(stats.malformed_samples) < 5:
                stats.malformed_samples.append(error)
            continue

        _accumulate_schema(record, config.required_post_fields, stats, config)

        month = _to_year_month(record.get("created_utc"))
        if month:
            monthly[month] += 1

        if stats.total % config.progress_interval == 0:
            log.info(f"  Posts: {stats.total:,} processed …")

    log.info(f"  Posts done — {stats.total:,} records, {stats.malformed_count} malformed lines")
    return stats, dict(monthly)


def validate_comments(config: Config) -> tuple[SchemaStats, ThreadStats, dict[str, int]]:
    path = config.data_dir / config.comments_file
    stats = SchemaStats(record_type="comments")
    thread_stats = ThreadStats()
    monthly: dict[str, int] = defaultdict(int)

    log.info(f"=== Scanning comments: {path.name} ({path.stat().st_size / 1_048_576:.0f} MB) ===")

    for _line_num, record, error in stream_ndjson(path):
        if error:
            stats.malformed_count += 1
            if len(stats.malformed_samples) < 5:
                stats.malformed_samples.append(error)
            continue

        _accumulate_schema(record, config.required_comment_fields, stats, config)
        _accumulate_thread(record, thread_stats, config)

        month = _to_year_month(record.get("created_utc"))
        if month:
            monthly[month] += 1

        if stats.total % config.progress_interval == 0:
            log.info(f"  Comments: {stats.total:,} processed …")

    log.info(f"  Comments done — {stats.total:,} records, {stats.malformed_count} malformed lines")
    return stats, thread_stats, dict(monthly)


# ── Console printing ────────────────────────────────────────────────────────────

def print_field_table(
    stats: SchemaStats,
    required_fields: tuple[str, ...],
) -> None:
    label = stats.record_type.upper()
    total = stats.total
    divider = "─" * 62

    print(f"\n{divider}")
    print(f"  {label} — {total:,} records  |  {stats.malformed_count} malformed lines")
    print(divider)
    print(f"  {'Field':<22} {'Present':>9} {'Present %':>11} {'[deleted]':>11}")
    print(f"  {'─'*20:<22} {'─'*7:>9} {'─'*9:>11} {'─'*9:>11}")

    for fname in required_fields:
        present = stats.field_present.get(fname, 0)
        deleted = stats.field_deleted.get(fname, 0)
        pct = (present / total * 100) if total > 0 else 0.0
        warn = "  ⚠️" if pct < 95.0 else ""
        print(f"  {fname:<22} {present:>9,} {pct:>10.1f}% {deleted:>11,}{warn}")

    unexpected = sorted(stats.all_field_names - set(required_fields))
    print(f"\n  Additional fields discovered: {len(unexpected)}")
    for fname in unexpected[:25]:
        print(f"    + {fname}")
    if len(unexpected) > 25:
        print(f"    … and {len(unexpected) - 25} more (see schema_report.json)")

    print(f"\n  Sample records (required fields only, first {len(stats.sample_records)}):")
    for i, rec in enumerate(stats.sample_records, start=1):
        print(f"  [{i}] {json.dumps(rec, ensure_ascii=False)[:200]}")


def print_thread_table(thread_stats: ThreadStats, total_comments: int) -> None:
    divider = "─" * 62
    safe_total = max(total_comments, 1)

    print(f"\n{divider}")
    print(f"  THREAD STRUCTURE ANALYSIS")
    print(divider)

    d_pct = thread_stats.direct_replies / safe_total * 100
    n_pct = thread_stats.nested_replies / safe_total * 100
    u_pct = thread_stats.unknown_parent / safe_total * 100
    l_pct = thread_stats.link_id_t3_count / safe_total * 100

    print(f"  Direct replies  (parent_id = t3_*)  {thread_stats.direct_replies:>9,}  ({d_pct:.1f}%)")
    print(f"  Nested replies  (parent_id = t1_*)  {thread_stats.nested_replies:>9,}  ({n_pct:.1f}%)")
    print(f"  Unknown parent                      {thread_stats.unknown_parent:>9,}  ({u_pct:.1f}%)")
    print(f"\n  link_id = t3_*  {thread_stats.link_id_t3_count:>9,}  ({l_pct:.1f}%)")
    print(f"  link_id = other {thread_stats.link_id_other_count:>9,}")

    print(f"\n  Direct reply samples (parent = t3_*):")
    for s in thread_stats.direct_samples:
        print(f"    id={s['id']}  parent={s['parent_id']}  link={s['link_id']}")
        print(f"    ↳ {s['body_preview']!r}")

    print(f"\n  Nested reply samples (parent = t1_*):")
    for s in thread_stats.nested_samples:
        print(f"    id={s['id']}  parent={s['parent_id']}  link={s['link_id']}")
        print(f"    ↳ {s['body_preview']!r}")


def print_coverage_table(
    post_monthly: dict[str, int],
    comment_monthly: dict[str, int],
    post_gaps: list[dict[str, Any]],
    comment_gaps: list[dict[str, Any]],
) -> None:
    divider = "─" * 62

    print(f"\n{divider}")
    print(f"  DATE COVERAGE  (monthly post + comment counts)")
    print(divider)
    print(f"  {'Month':<10}  {'Posts':>8}  {'Comments':>12}  {'Flag'}")
    print(f"  {'─'*8:<10}  {'─'*6:>8}  {'─'*10:>12}")

    flagged_months = {g["month"] for g in post_gaps + comment_gaps}
    all_months = sorted(set(post_monthly) | set(comment_monthly))

    for month in all_months:
        p = post_monthly.get(month, 0)
        c = comment_monthly.get(month, 0)
        flag = "  ⚠️" if month in flagged_months else ""
        print(f"  {month:<10}  {p:>8,}  {c:>12,}{flag}")

    if post_gaps or comment_gaps:
        print(f"\n  ⚠️  Flagged coverage issues:")
        for g in post_gaps:
            if g["reason"] == "large_drop":
                print(f"    Posts    {g['month']}: {g['drop_pct']}% drop vs {g['prior_month']}")
            else:
                print(f"    Posts    {g['month']}: zero records")
        for g in comment_gaps:
            if g["reason"] == "large_drop":
                print(f"    Comments {g['month']}: {g['drop_pct']}% drop vs {g['prior_month']}")
            else:
                print(f"    Comments {g['month']}: zero records")
    else:
        print(f"\n  ✅  No coverage gaps detected.")


# ── JSON report builders ────────────────────────────────────────────────────────

def _presence_pct(stats: SchemaStats, fields: tuple[str, ...]) -> dict[str, float]:
    if stats.total == 0:
        return {}
    return {
        f: round(stats.field_present.get(f, 0) / stats.total * 100, 2)
        for f in fields
    }


def build_schema_report(
    post_stats: SchemaStats,
    comment_stats: SchemaStats,
    thread_stats: ThreadStats,
    config: Config,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "posts": {
            "total_records": post_stats.total,
            "malformed_lines": post_stats.malformed_count,
            "malformed_samples": post_stats.malformed_samples,
            "field_presence_pct": _presence_pct(post_stats, config.required_post_fields),
            "field_deleted_counts": {
                f: post_stats.field_deleted.get(f, 0)
                for f in config.required_post_fields
            },
            "all_fields_discovered": sorted(post_stats.all_field_names),
            "unexpected_fields": sorted(
                post_stats.all_field_names - set(config.required_post_fields)
            ),
            "sample_records": post_stats.sample_records,
        },
        "comments": {
            "total_records": comment_stats.total,
            "malformed_lines": comment_stats.malformed_count,
            "malformed_samples": comment_stats.malformed_samples,
            "field_presence_pct": _presence_pct(comment_stats, config.required_comment_fields),
            "field_deleted_counts": {
                f: comment_stats.field_deleted.get(f, 0)
                for f in config.required_comment_fields
            },
            "all_fields_discovered": sorted(comment_stats.all_field_names),
            "unexpected_fields": sorted(
                comment_stats.all_field_names - set(config.required_comment_fields)
            ),
            "sample_records": comment_stats.sample_records,
        },
        "thread_structure": {
            "total_comments": comment_stats.total,
            "direct_replies_t3": thread_stats.direct_replies,
            "direct_replies_pct": round(
                thread_stats.direct_replies / max(comment_stats.total, 1) * 100, 2
            ),
            "nested_replies_t1": thread_stats.nested_replies,
            "nested_replies_pct": round(
                thread_stats.nested_replies / max(comment_stats.total, 1) * 100, 2
            ),
            "unknown_parent": thread_stats.unknown_parent,
            "link_id_t3_count": thread_stats.link_id_t3_count,
            "link_id_t3_pct": round(
                thread_stats.link_id_t3_count / max(comment_stats.total, 1) * 100, 2
            ),
            "link_id_other_count": thread_stats.link_id_other_count,
            "direct_reply_samples": thread_stats.direct_samples,
            "nested_reply_samples": thread_stats.nested_samples,
        },
    }


def build_coverage_report(
    post_monthly: dict[str, int],
    comment_monthly: dict[str, int],
    post_gaps: list[dict[str, Any]],
    comment_gaps: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "posts": {
            "monthly_counts": dict(sorted(post_monthly.items())),
            "flagged_months": post_gaps,
        },
        "comments": {
            "monthly_counts": dict(sorted(comment_monthly.items())),
            "flagged_months": comment_gaps,
        },
    }


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    config = Config()

    posts_path = config.data_dir / config.posts_file
    comments_path = config.data_dir / config.comments_file

    for path in (posts_path, comments_path):
        if not path.exists():
            log.error(f"Data file not found: {path}")
            sys.exit(1)

    # ── Tasks 1 + 2: Posts ─────────────────────────────────────────────────
    post_stats, post_monthly = validate_posts(config)

    # ── Tasks 1 + 2: Comments (single pass: schema + thread + coverage) ────
    comment_stats, thread_stats, comment_monthly = validate_comments(config)

    # ── Gap detection ───────────────────────────────────────────────────────
    post_gaps = detect_gaps(post_monthly, config.coverage_drop_threshold)
    comment_gaps = detect_gaps(comment_monthly, config.coverage_drop_threshold)

    # ── Console output ──────────────────────────────────────────────────────
    divider = "═" * 62
    print(f"\n{divider}")
    print(f"  SCHEMA VALIDATION REPORT — Long COVID Reddit RAG")
    print(divider)

    print_field_table(post_stats, config.required_post_fields)
    print_field_table(comment_stats, config.required_comment_fields)
    print_thread_table(thread_stats, comment_stats.total)
    print_coverage_table(post_monthly, comment_monthly, post_gaps, comment_gaps)

    # ── Write JSON reports ──────────────────────────────────────────────────
    config.reports_dir.mkdir(parents=True, exist_ok=True)

    schema_out = config.reports_dir / config.schema_report_out
    schema_report = build_schema_report(post_stats, comment_stats, thread_stats, config)
    schema_out.write_text(json.dumps(schema_report, indent=2, ensure_ascii=False))
    log.info(f"Schema report → {schema_out}")

    coverage_out = config.reports_dir / config.coverage_report_out
    coverage_report = build_coverage_report(
        post_monthly, comment_monthly, post_gaps, comment_gaps
    )
    coverage_out.write_text(json.dumps(coverage_report, indent=2, ensure_ascii=False))
    log.info(f"Coverage report → {coverage_out}")

    print(f"\n✅  Validation complete.\n")


if __name__ == "__main__":
    main()
