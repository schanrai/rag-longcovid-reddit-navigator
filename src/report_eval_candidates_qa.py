#!/usr/bin/env python3
"""
report_eval_candidates_qa.py — Human QA: full text for every golden-query → candidate chunk_id.

Reads data/golden_queries.json + data/eval_corpus_positives.json, resolves each chunk_id
from comment_chunks.jsonl and post_chunks.jsonl (full corpus, not only eval_corpus),
and writes a Markdown file with complete post_title, post_summary, and text per candidate.

Usage:
  python3 src/report_eval_candidates_qa.py
  python3 src/report_eval_candidates_qa.py --out reports/eval_candidates_qa.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator


def stream_chunks(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _md_fence(s: str) -> str:
    """Escape so content can live inside a ```text fence."""
    if not s:
        return "_(_empty_)_"
    t = s.replace("\r\n", "\n").replace("\r", "\n")
    t = t.replace("```", "``\u200b`")  # zero-width space so fence won't break
    return t


def load_chunk_records(
    *,
    needed: set[str],
    comment_path: Path,
    post_path: Path,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    remaining = set(needed)

    for path in (comment_path, post_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing chunk file: {path}")
        for rec in stream_chunks(path):
            cid = rec.get("chunk_id")
            if not cid or cid not in remaining:
                continue
            out[str(cid)] = rec
            remaining.discard(cid)
            if not remaining:
                return out

    if remaining:
        missing = sorted(remaining)[:15]
        raise ValueError(
            f"{len(remaining)} chunk_id(s) not found in chunk JSONL. Examples: {missing}"
        )
    return out


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="Write Markdown QA report for eval candidate chunks.")
    ap.add_argument("--golden", type=Path, default=root / "data" / "golden_queries.json")
    ap.add_argument(
        "--positives",
        type=Path,
        default=root / "data" / "eval_corpus_positives.json",
    )
    ap.add_argument("--comment-chunks", type=Path, default=root / "data" / "comment_chunks.jsonl")
    ap.add_argument("--post-chunks", type=Path, default=root / "data" / "post_chunks.jsonl")
    ap.add_argument("--out", type=Path, default=root / "reports" / "eval_candidates_qa.md")
    args = ap.parse_args()

    if not args.golden.exists():
        print(f"Missing {args.golden}", file=sys.stderr)
        sys.exit(1)
    if not args.positives.exists():
        print(f"Missing {args.positives} — run build_eval_corpus.py first", file=sys.stderr)
        sys.exit(1)

    gold_raw = json.loads(args.golden.read_text(encoding="utf-8"))
    queries = {str(q["id"]): q for q in gold_raw["queries"]}

    pos_raw = json.loads(args.positives.read_text(encoding="utf-8"))
    qmap: dict[str, list[str]] = pos_raw["query_id_to_candidate_positives"]

    needed: set[str] = set()
    for ids in qmap.values():
        needed.update(ids)

    print(f"Resolving {len(needed):,} unique chunk_ids from corpus…", file=sys.stderr)
    by_id = load_chunk_records(
        needed=needed,
        comment_path=args.comment_chunks,
        post_path=args.post_chunks,
    )

    lines: list[str] = []
    lines.append("# Eval candidate chunks — full text QA report")
    lines.append("")
    lines.append(
        "Auto-generated from `golden_queries.json` + `eval_corpus_positives.json`. "
        "Each block is the **full** stored chunk (title, summary, text)."
    )
    lines.append("")

    for qid in sorted(qmap.keys(), key=lambda x: (len(x), x)):
        q = queries.get(qid, {})
        lines.append("---")
        lines.append("")
        lines.append(f"## {qid} — {q.get('category', '')}")
        lines.append("")
        lines.append(f"**Query:** {q.get('query', '')}")
        lines.append("")
        lines.append(f"**Notes:** {q.get('notes', '')}")
        lines.append("")
        terms = q.get("expected_terms") or []
        lines.append(f"**Expected terms:** {', '.join(str(t) for t in terms)}")
        lines.append("")

        cids = qmap[qid]
        for i, cid in enumerate(cids, 1):
            rec = by_id.get(cid)
            lines.append(f"### Candidate {i}/{len(cids)} — `{cid}`")
            lines.append("")
            if rec is None:
                lines.append("_Chunk record not found._")
                lines.append("")
                continue

            src = "comment" if cid.startswith("t1_") else "post"
            lines.append(f"**Source:** {src}-chunk")
            lines.append("")
            title = rec.get("post_title") or ""
            lines.append(f"**post_title:** {title}")
            lines.append("")
            summ = rec.get("post_summary")
            if summ is not None and str(summ).strip():
                lines.append("**post_summary:**")
                lines.append("")
                lines.append("```text")
                lines.append(_md_fence(str(summ)))
                lines.append("```")
                lines.append("")
            body = rec.get("text") or ""
            lines.append("**text (full):**")
            lines.append("")
            lines.append("```text")
            lines.append(_md_fence(str(body)))
            lines.append("```")
            lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out} ({len(lines):,} lines)", file=sys.stderr)


if __name__ == "__main__":
    main()
