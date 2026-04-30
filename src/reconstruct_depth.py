#!/usr/bin/env python3
"""
reconstruct_depth.py — Step 1: reconstruct true comment depth for chunk corpus.

Reads:
  - data/comment_chunks_pre_enrich.jsonl (chunked comments)
  - data/r_LongCovid_comments.jsonl (raw comments for parent chains + parent text)

Writes:
  - data/comment_chunks_with_depth.jsonl (versioned output)
  - reports/reconstruct_depth_report.json (QA + integrity metrics)

Usage:
  python3 src/reconstruct_depth.py
  python3 src/reconstruct_depth.py --help
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("reconstruct_depth")

SCHEMA_REPORT: Final[str] = "reconstruct_depth_report_v1"
MAX_PARENT_SENTENCE_CHARS: Final[int] = 280


def stream_ndjson(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line), None
            except json.JSONDecodeError as exc:
                yield line_no, None, exc


def tokenize_word_count(text: str) -> int:
    return len(text.split())


def first_sentence(text: str) -> str:
    """
    Conservative first-sentence extractor.
    Falls back to the first line/chars if no sentence punctuation is found.
    """
    s = " ".join((text or "").strip().split())
    if not s:
        return ""
    for punct in (". ", "! ", "? "):
        idx = s.find(punct)
        if idx != -1:
            return s[: idx + 1].strip()[:MAX_PARENT_SENTENCE_CHARS]
    return s[:MAX_PARENT_SENTENCE_CHARS].strip()


@dataclass(frozen=True)
class RawComment:
    parent_id: str
    body: str
    score: int


@dataclass
class Counters:
    total_chunk_rows: int = 0
    chunk_rows_with_t1_parent: int = 0
    chunk_rows_with_t3_parent: int = 0
    chunk_rows_with_other_parent: int = 0
    chunk_rows_with_reconstructed_depth: int = 0
    chunk_rows_missing_depth: int = 0
    missing_parent_links: int = 0
    missing_parent_in_raw: int = 0
    cycle_detected: int = 0
    malformed_parent: int = 0
    parent_sentence_present: int = 0
    parent_sentence_missing: int = 0


class DepthResolver:
    def __init__(self, raw_comments: dict[str, RawComment], counters: Counters) -> None:
        self.raw_comments = raw_comments
        self.counters = counters
        self.memo_depth: dict[str, int | None] = {}

    def resolve_depth(self, comment_id: str) -> int | None:
        """
        Depth convention:
          - direct reply to post (parent t3_*) => depth 1
          - reply to reply increases by 1
        Returns None when chain cannot be resolved safely.
        """
        if comment_id in self.memo_depth:
            return self.memo_depth[comment_id]

        visited: set[str] = set()
        chain: list[str] = []
        cur = comment_id

        while True:
            if cur in self.memo_depth:
                base = self.memo_depth[cur]
                if base is None:
                    for cid in chain:
                        self.memo_depth[cid] = None
                    return None
                depth = base
                for cid in reversed(chain):
                    depth += 1
                    self.memo_depth[cid] = depth
                return self.memo_depth.get(comment_id)

            if cur in visited:
                self.counters.cycle_detected += 1
                for cid in chain:
                    self.memo_depth[cid] = None
                return None

            visited.add(cur)
            chain.append(cur)
            rec = self.raw_comments.get(cur)
            if rec is None:
                self.counters.missing_parent_in_raw += 1
                for cid in chain:
                    self.memo_depth[cid] = None
                return None

            parent = rec.parent_id
            if parent.startswith("t3_"):
                depth = 1
                for cid in reversed(chain):
                    self.memo_depth[cid] = depth
                    depth += 1
                return self.memo_depth.get(comment_id)

            if not parent.startswith("t1_"):
                self.counters.malformed_parent += 1
                for cid in chain:
                    self.memo_depth[cid] = None
                return None

            cur = parent.split("_", 1)[1]


def load_raw_comments(path: Path) -> dict[str, RawComment]:
    out: dict[str, RawComment] = {}
    bad = 0
    for _line_no, rec, err in stream_ndjson(path):
        if err or rec is None:
            bad += 1
            continue
        cid = str(rec.get("id") or "")
        if not cid:
            continue
        out[cid] = RawComment(
            parent_id=str(rec.get("parent_id") or ""),
            body=str(rec.get("body") or ""),
            score=int(rec.get("score") or 0),
        )
    if bad:
        log.warning("Skipped %s malformed raw-comment JSON lines", bad)
    log.info("Loaded raw comments: %s", f"{len(out):,}")
    return out


def reconstruct(
    *,
    chunk_in: Path,
    chunk_out: Path,
    raw_comments_input: Path,
    raw_comments: dict[str, RawComment],
    report_out: Path,
    sample_size: int,
    seed: int,
) -> None:
    counters = Counters()
    resolver = DepthResolver(raw_comments=raw_comments, counters=counters)
    depth_hist: dict[str, int] = {
        "unknown": 0,
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 0,
        "4+": 0,
    }
    sample_pool: list[dict[str, Any]] = []

    chunk_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)

    with chunk_in.open(encoding="utf-8") as fin, chunk_out.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            counters.total_chunk_rows += 1

            chunk_id = str(rec.get("chunk_id") or "")
            parent_id = str(rec.get("parent_id") or "")
            depth: int | None = None
            parent_snip = ""

            if chunk_id.startswith("t1_"):
                parts = chunk_id.split("_")
                if len(parts) >= 3:
                    comment_id = parts[1]
                    depth = resolver.resolve_depth(comment_id)
                if parent_id.startswith("t1_"):
                    counters.chunk_rows_with_t1_parent += 1
                    parent_comment_id = parent_id.split("_", 1)[1]
                    parent_rec = raw_comments.get(parent_comment_id)
                    if parent_rec is None:
                        counters.missing_parent_links += 1
                    else:
                        parent_snip = first_sentence(parent_rec.body)
                        if parent_snip:
                            counters.parent_sentence_present += 1
                        else:
                            counters.parent_sentence_missing += 1
                elif parent_id.startswith("t3_"):
                    counters.chunk_rows_with_t3_parent += 1
                else:
                    counters.chunk_rows_with_other_parent += 1
            else:
                depth = int(rec.get("nest_level") or 0)

            if depth is None:
                counters.chunk_rows_missing_depth += 1
                depth_hist["unknown"] += 1
            else:
                counters.chunk_rows_with_reconstructed_depth += 1
                if depth <= 0:
                    depth_hist["0"] += 1
                elif depth == 1:
                    depth_hist["1"] += 1
                elif depth == 2:
                    depth_hist["2"] += 1
                elif depth == 3:
                    depth_hist["3"] += 1
                else:
                    depth_hist["4+"] += 1

            rec["nest_level_original"] = rec.get("nest_level")
            rec["nest_level"] = depth if depth is not None else rec.get("nest_level")
            rec["nest_level_reconstructed"] = depth
            rec["depth_source"] = "raw_parent_chain_v1" if chunk_id.startswith("t1_") else "post_chunk"
            rec["parent_first_sentence"] = parent_snip if chunk_id.startswith("t1_") else ""

            if chunk_id.startswith("t1_") and len(sample_pool) < 5000:
                sample_pool.append(
                    {
                        "chunk_id": chunk_id,
                        "parent_id": parent_id,
                        "nest_level_reconstructed": rec["nest_level_reconstructed"],
                        "parent_first_sentence": rec["parent_first_sentence"],
                        "text_preview": str(rec.get("text") or "")[:140],
                    }
                )

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    rng = random.Random(seed)
    samples = rng.sample(sample_pool, k=min(sample_size, len(sample_pool))) if sample_pool else []

    payload = {
        "schema": SCHEMA_REPORT,
        "inputs": {
            "chunk_in": str(chunk_in),
            "raw_comments": str(raw_comments_input),
        },
        "outputs": {
            "chunk_out": str(chunk_out),
            "report_out": str(report_out),
        },
        "counters": counters.__dict__,
        "depth_histogram": depth_hist,
        "qa_samples": samples,
    }
    report_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info("Wrote reconstructed chunks: %s", chunk_out)
    log.info("Wrote reconstruction report: %s", report_out)
    log.info("Depth histogram: %s", depth_hist)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="Reconstruct true comment depth from raw parent chains.")
    ap.add_argument(
        "--chunk-in",
        type=Path,
        default=root / "data" / "comment_chunks_pre_enrich.jsonl",
        help="Input chunked comments file (pre-enrichment baseline).",
    )
    ap.add_argument(
        "--raw-comments",
        type=Path,
        default=root / "data" / "r_LongCovid_comments.jsonl",
        help="Raw comments dump used for parent-chain walking.",
    )
    ap.add_argument(
        "--chunk-out",
        type=Path,
        default=root / "data" / "comment_chunks_with_depth.jsonl",
        help="Versioned output with reconstructed depth fields.",
    )
    ap.add_argument(
        "--report-out",
        type=Path,
        default=root / "reports" / "reconstruct_depth_report.json",
        help="QA/integrity report output path.",
    )
    ap.add_argument("--sample-size", type=int, default=30, help="Number of QA samples in report.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    chunk_in_path = args.chunk_in.resolve()
    raw_comments_path = args.raw_comments.resolve()
    chunk_out_path = args.chunk_out.resolve()
    report_out_path = args.report_out.resolve()

    if not chunk_in_path.exists():
        raise FileNotFoundError(f"Missing chunk input: {chunk_in_path}")
    if not raw_comments_path.exists():
        raise FileNotFoundError(f"Missing raw comments: {raw_comments_path}")

    raw_map = load_raw_comments(raw_comments_path)
    reconstruct(
        chunk_in=chunk_in_path,
        chunk_out=chunk_out_path,
        raw_comments_input=raw_comments_path,
        raw_comments=raw_map,
        report_out=report_out_path,
        sample_size=args.sample_size,
        seed=args.seed,
    )
