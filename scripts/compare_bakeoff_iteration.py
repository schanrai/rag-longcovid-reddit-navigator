#!/usr/bin/env python3
"""
Compare synthesis eval: bake-off JSON (per-model slice) vs single-model iteration JSON.

Typical use (Gemini 3 slice from bakeoff_2 vs a new iteration on the same five queries):

  cd projects/rag-longcovid-reddit-navigator
  python scripts/compare_bakeoff_iteration.py \\
    --bakeoff reports/synthesis_eval/bakeoff_2.json \\
    --iteration reports/synthesis_eval/iteration_4.json

Requires matching `query` strings in the same order between the two files.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

AGG_KEYS = [
    "instruction_adherence",
    "citation_accuracy",
    "format_consistency",
    "tone_intent",
    "diversity",
    "mean",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve(p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _short(text: str, max_len: int = 44) -> str:
    t = text.replace("\n", " ").strip()
    return (t[: max_len - 1] + "…") if len(t) > max_len else t


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compare bake-off per-model scores vs a single-model iteration JSON.",
    )
    ap.add_argument(
        "--bakeoff",
        type=str,
        default="reports/synthesis_eval/bakeoff_2.json",
        help="Path to bakeoff_*.json (run_type=bakeoff)",
    )
    ap.add_argument(
        "--iteration",
        type=str,
        default="reports/synthesis_eval/iteration_4.json",
        help="Path to iteration_*.json (run_type=single)",
    )
    ap.add_argument(
        "--model-id",
        type=str,
        default="google/gemini-3-flash-preview",
        help="OpenRouter model id key under queries[].by_model",
    )
    args = ap.parse_args()

    bake_path = _resolve(args.bakeoff)
    iter_path = _resolve(args.iteration)

    if not bake_path.is_file():
        print(f"error: bakeoff file not found: {bake_path}", file=sys.stderr)
        return 1
    if not iter_path.is_file():
        print(f"error: iteration file not found: {iter_path}", file=sys.stderr)
        return 1

    bake = json.loads(bake_path.read_text(encoding="utf-8"))
    it = json.loads(iter_path.read_text(encoding="utf-8"))

    if bake.get("run_type") != "bakeoff":
        print("error: --bakeoff JSON must have run_type=bakeoff", file=sys.stderr)
        return 1
    if it.get("run_type") != "single":
        print("error: --iteration JSON must have run_type=single", file=sys.stderr)
        return 1

    mid = args.model_id
    if mid not in bake.get("aggregate_by_model", {}):
        print(f"error: model id not in aggregate_by_model: {mid!r}", file=sys.stderr)
        print(f"  keys: {list(bake.get('aggregate_by_model', {}).keys())}", file=sys.stderr)
        return 1

    b_agg = bake["aggregate_by_model"][mid]
    i_agg = it["aggregate"]

    print("=== Paths ===")
    print(f"bakeoff:   {bake_path}")
    print(f"iteration: {iter_path}")
    print(f"model_id:  {mid}")
    print()
    print("=== Metadata ===")
    print(f"bakeoff timestamp: {bake.get('timestamp', '')}  judge: {bake.get('judge_model', '')}")
    print(f"iteration ts:      {it.get('timestamp', '')}  judge: {it.get('judge_model', '')}")
    print(f"iteration synth:   {it.get('synthesis_model', '')}")
    print()

    hdr = f"{'criterion':<26}  {'bakeoff':>12}  {'iteration':>12}  {'delta':>8}"
    print("=== Aggregate ===")
    print(hdr)
    print("-" * len(hdr))
    for k in AGG_KEYS:
        b = b_agg[k]
        i = i_agg[k]
        delta = round(float(i) - float(b), 1)
        print(f"{k:<26}  {b:>12}  {i:>12}  {delta:>8}")
    print()

    b_queries = bake["queries"]
    i_queries = it["queries"]
    if len(b_queries) != len(i_queries):
        print("error: query count mismatch", len(b_queries), "vs", len(i_queries), file=sys.stderr)
        return 1

    print("=== Per-query (citation_accuracy, mean, issue counts) ===")
    col = f"{'#':>2}  {'query':<44}  {'b cite':>7}  {'i cite':>7}  {'Δ':>5}  {'b mean':>7}  {'i mean':>7}  {'b #i':>5}  {'i #i':>5}"
    print(col)
    print("-" * len(col))

    for idx, (bq, iq) in enumerate(zip(b_queries, i_queries), start=1):
        if bq.get("query") != iq.get("query"):
            print("error: query text mismatch at index", idx, file=sys.stderr)
            print("  bakeoff: ", repr(bq.get("query")), file=sys.stderr)
            print("  iter:    ", repr(iq.get("query")), file=sys.stderr)
            return 1
        if mid not in bq.get("by_model", {}):
            print(f"error: missing by_model[{mid!r}] for query index {idx}", file=sys.stderr)
            return 1
        bm = bq["by_model"][mid]
        if bm.get("error"):
            print(f"error: bakeoff model has error for query {idx}: {bm.get('error')}", file=sys.stderr)
            return 1
        bsc = bm["scores"]
        isc = iq["scores"]
        b_issues = len(bm.get("issues") or [])
        i_issues = len(iq.get("issues") or [])
        dc = int(isc["citation_accuracy"]) - int(bsc["citation_accuracy"])
        dm = round(float(isc["mean"]) - float(bsc["mean"]), 1)
        print(
            f"{idx:>2}  {_short(bq['query'], 44):<44}  "
            f"{bsc['citation_accuracy']:>7}  {isc['citation_accuracy']:>7}  {dc:>5}  "
            f"{bsc['mean']:>7}  {isc['mean']:>7}  {b_issues:>5}  {i_issues:>5}"
        )

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
