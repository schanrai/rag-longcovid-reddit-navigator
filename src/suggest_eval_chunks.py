#!/usr/bin/env python3
"""
suggest_eval_chunks.py — Phase 1b: surface candidate chunks for golden-query labeling.

Scans comment_chunks.jsonl and post_chunks.jsonl once. For each golden query:
  - Counts expected_term hits in post_title, post_summary, and text (case-insensitive).
  - Short / acronym terms use word-boundary matching to reduce false positives.
  - Queries with category "emotional" also match sentiment-style patterns on post_title only.
  - Queries with category "benefits" also match disability/SSDI-style patterns on post_title.

Writes a Markdown report for human review. After you pick chunk_ids and relevance (1–3),
merge them into data/golden_queries.json as labeled_relevant, then run build_eval_corpus.py
(when implemented).

Usage:
  python3 src/suggest_eval_chunks.py
  python3 src/suggest_eval_chunks.py --max-per-query 30 --out reports/eval_candidate_report.md
"""
from __future__ import annotations

import argparse
import heapq
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

# ── Sentiment / topic hints (post_title only) ─────────────────────────────────

_EMOTIONAL_TITLE = re.compile(
    r"(?i)(miss\b|old\s+me\b|grief|who\s+i\s+was|forgotten|ruined\s+my|"
    r"hate\s|sucks?\b|don'?t\s+believe|gaslight|alone\b|depress|anxiety\b|"
    r"identity|not\s+the\s+same|lost\s+myself|no\s+respect\s+for)"
)

_BENEFITS_TITLE = re.compile(
    r"(?i)(disability|ssdi|ssa\b|denied\b|appeal\b|benefits\b|"
    r"unable\s+to\s+work|monthly\s+check|long[-\s]?term\s+disability)"
)


@dataclass(frozen=True)
class Config:
    data_dir: Path
    golden_path: Path
    comment_chunks: Path
    post_chunks: Path
    out_path: Path
    max_per_query: int
    text_preview_len: int


def load_queries(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return list(raw["queries"])


def stream_chunks(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _haystack(rec: dict[str, Any]) -> str:
    parts = [
        (rec.get("post_title") or ""),
        str(rec.get("post_summary") or ""),
        (rec.get("text") or ""),
    ]
    return "\n".join(parts).lower()


def _title_only(rec: dict[str, Any]) -> str:
    return (rec.get("post_title") or "").lower()


def term_match(term: str, haystack_lower: str) -> bool:
    """
    Match expected_term against lowercased haystack.
    Short tokens and acronyms use word boundaries.
    """
    t = term.strip()
    if not t:
        return False
    tl = t.lower()
    hay = haystack_lower

    if " " in t:
        return tl in hay

    # Single token: boundary if short or all-caps style acronym in golden file
    if len(t) <= 4 or t.isupper():
        pat = r"(?<![a-z0-9])" + re.escape(tl) + r"(?![a-z0-9])"
        return re.search(pat, hay, re.IGNORECASE) is not None

    return tl in hay


def count_term_hits(terms: list[str], haystack_lower: str) -> int:
    return sum(1 for term in terms if term_match(term, haystack_lower))


def chunk_vote(rec: dict[str, Any]) -> int:
    if "comment_score" in rec:
        return int(rec.get("comment_score") or 0)
    return int(rec.get("post_score") or 0)


def chunk_source_label(rec: dict[str, Any]) -> str:
    return "comment" if "comment_id" in rec else "post"


@dataclass
class CandidateRow:
    score: int
    chunk_id: str
    source: str
    post_title: str
    text_preview: str
    vote: int
    term_hits: int
    title_hint: str  # "" | "emotional_title" | "benefits_title"

    def __lt__(self, other: CandidateRow) -> bool:
        if self.score != other.score:
            return self.score < other.score
        return self.chunk_id > other.chunk_id


def consider_queries_for_chunk(
    queries: list[dict[str, Any]],
    rec: dict[str, Any],
    tops: dict[str, list[CandidateRow]],
    k: int,
    preview_len: int,
) -> None:
    hay = _haystack(rec)
    title_raw = rec.get("post_title") or ""
    title_l = title_raw.lower()
    chunk_id = rec.get("chunk_id") or ""
    if not chunk_id:
        return

    src = chunk_source_label(rec)
    vote = chunk_vote(rec)
    text = (rec.get("text") or "").replace("\n", " ")
    preview = text[:preview_len] + ("…" if len(text) > preview_len else "")
    title_display = title_raw[:200]

    for q in queries:
        qid = q.get("id")
        if not qid:
            continue

        terms = list(q.get("expected_terms") or [])
        raw_hits = count_term_hits(terms, hay)
        th = raw_hits
        cat = q.get("category") or ""
        hint = ""
        if cat == "emotional" and _EMOTIONAL_TITLE.search(title_raw):
            if raw_hits == 0:
                th = 1
                hint = "emotional_title"
            else:
                th = max(th, 1)
        if cat == "benefits" and _BENEFITS_TITLE.search(title_raw):
            if raw_hits == 0:
                th = 1
                hint = "benefits_title"
            else:
                th = max(th, 1)

        if th == 0:
            continue

        score = th * 10_000 + vote
        row = CandidateRow(
            score=score,
            chunk_id=chunk_id,
            source=src,
            post_title=title_display,
            text_preview=preview,
            vote=vote,
            term_hits=th,
            title_hint=hint,
        )
        bucket = tops.setdefault(str(qid), [])
        if len(bucket) < k:
            heapq.heappush(bucket, row)
        elif score > bucket[0].score:
            heapq.heapreplace(bucket, row)


def run(cfg: Config) -> dict[str, Any]:
    queries = load_queries(cfg.golden_path)
    tops: dict[str, list[CandidateRow]] = {}

    for path in (cfg.comment_chunks, cfg.post_chunks):
        if not path.exists():
            raise FileNotFoundError(f"Missing chunk file: {path}")
        n = 0
        for rec in stream_chunks(path):
            consider_queries_for_chunk(queries, rec, tops, cfg.max_per_query, cfg.text_preview_len)
            n += 1
            if n % 50_000 == 0:
                print(f"  … scanned {n:,} lines from {path.name}", file=sys.stderr)
        print(f"  Done {path.name}: {n:,} chunks", file=sys.stderr)

    # Build sorted lists (descending score)
    resolved: dict[str, list[CandidateRow]] = {}
    for q in queries:
        qid = str(q.get("id") or "")
        heap = tops.get(qid, [])
        resolved[qid] = sorted(heap, key=lambda r: r.score, reverse=True)

    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Eval candidate chunks (Phase 1b)")
    lines.append("")
    lines.append(f"Generated: `{datetime.now(tz=timezone.utc).isoformat()}`")
    lines.append("")
    lines.append("## How to use")
    lines.append("")
    lines.append(
        "This report is a reference for understanding what chunks exist per query category. "
        "The top-ranked candidates (by term hits + score) are used as **auto-selected positives** "
        "in `build_eval_corpus.py` — no manual labeling required."
    )
    lines.append("")
    lines.append(
        "To build the eval corpus: `python3 src/build_eval_corpus.py` "
        "(reads this manifest + chunk JSONL files, adds random distractors, writes `data/eval_corpus.jsonl`)."
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    summary_counts: dict[str, int] = {}

    for q in queries:
        qid = str(q.get("id") or "")
        cat = q.get("category") or ""
        qu = q.get("query") or ""
        rows = resolved.get(qid, [])
        summary_counts[qid] = len(rows)

        lines.append(f"## {qid} — {cat}")
        lines.append("")
        lines.append(f"**Query:** {qu}")
        lines.append("")
        lines.append(f"**Candidates:** {len(rows)} (max {cfg.max_per_query})")
        lines.append("")

        if not rows:
            lines.append("*No candidates matched. Broaden `expected_terms` or add title patterns.*")
            lines.append("")
            continue

        lines.append(
            "| chunk_id | src | vote | hits | hint | post_title | text_preview |"
        )
        lines.append(
            "|----------|-----|------|------|------|------------|--------------|"
        )
        for r in rows:
            hint = r.title_hint or "—"
            title_esc = r.post_title.replace("|", "\\|")
            prev_esc = r.text_preview.replace("|", "\\|")
            lines.append(
                f"| `{r.chunk_id}` | {r.source} | {r.vote} | {r.term_hits} | {hint} | "
                f"{title_esc} | {prev_esc} |"
            )
        lines.append("")

    manifest = {
        "schema": "eval_candidate_report_manifest_v1",
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "max_per_query": cfg.max_per_query,
        "queries": [
            {
                "id": q.get("id"),
                "category": q.get("category"),
                "candidate_count": summary_counts.get(str(q.get("id")), 0),
            }
            for q in queries
        ],
    }
    manifest_path = cfg.out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    cfg.out_path.write_text("\n".join(lines), encoding="utf-8")
    return manifest


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="Suggest eval chunks for golden queries (Markdown report).")
    ap.add_argument(
        "--golden",
        type=Path,
        default=root / "data" / "golden_queries.json",
        help="Path to golden_queries.json",
    )
    ap.add_argument(
        "--comment-chunks",
        type=Path,
        default=root / "data" / "comment_chunks.jsonl",
    )
    ap.add_argument(
        "--post-chunks",
        type=Path,
        default=root / "data" / "post_chunks.jsonl",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=root / "reports" / "eval_candidate_report.md",
    )
    ap.add_argument("--max-per-query", type=int, default=25)
    ap.add_argument("--preview-len", type=int, default=180)
    args = ap.parse_args()

    cfg = Config(
        data_dir=root / "data",
        golden_path=args.golden,
        comment_chunks=args.comment_chunks,
        post_chunks=args.post_chunks,
        out_path=args.out,
        max_per_query=args.max_per_query,
        text_preview_len=args.preview_len,
    )

    if not cfg.golden_path.exists():
        print(f"Missing {cfg.golden_path}", file=sys.stderr)
        sys.exit(1)

    print("Scanning corpus for golden-query candidates…", file=sys.stderr)
    run(cfg)
    print(f"Wrote {cfg.out_path}", file=sys.stderr)
    print(f"Wrote {cfg.out_path.with_suffix('.manifest.json')}", file=sys.stderr)


if __name__ == "__main__":
    main()
