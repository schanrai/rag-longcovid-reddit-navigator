#!/usr/bin/env python3
"""
eval_synthesis_golden.py — Full golden set: load all queries from golden_queries.json,
run retrieve → synthesis → LLM judge per query.

The fixed five-query harness remains ``python -m src.eval_synthesis`` (EVAL_QUERIES in
eval_synthesis_common.py) for bake-offs and regression comparisons. This module is for
Phase 5-style coverage over the canonical JSON file only.

Usage:
    cd projects/rag-longcovid-reddit-navigator
    python -m src.eval_synthesis_golden
    python -m src.eval_synthesis_golden --golden data/golden_queries.json

Writes: reports/synthesis_eval/golden_iteration_<N>.json (N auto-increments; separate from
iteration_<N>.json from eval_synthesis.py).

Requires: OPENROUTER_API_KEY, VOYAGE_API_KEY, Weaviate env (same as pipeline_cli).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.eval_synthesis_common import (
    AGG_KEYS,
    JUDGE_MODEL,
    PROJECT_ROOT,
    SYNTH_MODEL,
    call_judge,
    default_judge_result,
    judge_user_payload,
    normalize_scores,
    query_label_short,
    synthesis_telemetry,
)
from src.retrieval.config import RetrievalConfig
from src.retrieval.hybrid_search import _build_weaviate_client
from src.retrieval.pipeline import retrieve
from src.synthesis import SynthesisConfig, generate_synthesis, pack_context

log = logging.getLogger("eval_synthesis_golden")


def next_golden_iteration_index(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    best = 0
    for p in out_dir.glob("golden_iteration_*.json"):
        m = re.match(r"golden_iteration_(\d+)\.json$", p.name)
        if m:
            best = max(best, int(m.group(1)))
    return best + 1


def load_golden_query_rows(path: Path) -> tuple[list[dict[str, Any]], Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    queries = raw.get("queries")
    if not isinstance(queries, list) or not queries:
        raise ValueError(f"{path}: missing non-empty 'queries' array")
    rows: list[dict[str, Any]] = []
    for item in queries:
        if not isinstance(item, dict):
            continue
        qid = str(item.get("id", "")).strip()
        qtext = str(item.get("query", "")).strip()
        if not qtext:
            log.warning("Skipping entry with empty query: id=%r", qid)
            continue
        rows.append(
            {
                "id": qid,
                "query": qtext,
                "category": str(item.get("category", "")).strip(),
            }
        )
    if not rows:
        raise ValueError(f"{path}: no usable queries after load")
    return rows, raw.get("version")


def _run_golden_iteration(*, golden_path: Path) -> Path:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    voyage_key = os.environ.get("VOYAGE_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    if not voyage_key:
        raise SystemExit("VOYAGE_API_KEY is not set")

    golden_rows, golden_version = load_golden_query_rows(golden_path)
    syn_base = SynthesisConfig(model=SYNTH_MODEL)
    retrieval_cfg = RetrievalConfig()
    retrieval_cfg.reranker.enabled = True
    max_chunks = syn_base.max_chunks_in_context

    out_dir = PROJECT_ROOT / "reports" / "synthesis_eval"
    iteration = next_golden_iteration_index(out_dir)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    queries_out: list[dict[str, Any]] = []
    score_matrix: list[list[float]] = []

    client = _build_weaviate_client()
    try:
        for i, row in enumerate(golden_rows, start=1):
            query = row["query"]
            log.info(
                "Query %d/%d [%s]: %s",
                i,
                len(golden_rows),
                row["id"] or "?",
                query[:60],
            )
            retrieval = retrieve(
                query,
                cfg=retrieval_cfg,
                voyage_api_key=voyage_key,
                weaviate_client=client,
            )
            syn_resp = generate_synthesis(retrieval, cfg=syn_base)
            ctx_results = retrieval.results[:max_chunks]
            packed = pack_context(ctx_results, max_chunks=max_chunks)

            judge_user = judge_user_payload(
                original_query=retrieval.query.original_query,
                rewritten_query=retrieval.query.best_rewrite.query,
                answer_markdown=syn_resp.answer,
                packed_context=packed,
            )
            try:
                judge_raw = call_judge(judge_user, api_key=api_key)
                scores = normalize_scores(judge_raw)
            except Exception as exc:
                log.exception("Judge failed for golden_id=%s", row["id"])
                scores = default_judge_result(str(exc))

            row_scores = [float(scores[k]) for k in AGG_KEYS]
            score_matrix.append(row_scores)

            q_mean = round(sum(row_scores) / len(row_scores), 1) if row_scores else 0.0
            queries_out.append(
                {
                    "golden_id": row["id"],
                    "category": row["category"],
                    "query": query,
                    "rewritten_query": retrieval.query.best_rewrite.query,
                    "answer_markdown": syn_resp.answer,
                    "sources_cited": len(syn_resp.sources),
                    "sources_provided": len(ctx_results),
                    "synthesis": synthesis_telemetry(syn_resp),
                    "scores": {
                        **{k: scores[k] for k in AGG_KEYS},
                        "mean": q_mean,
                    },
                    "issues": scores["issues"],
                    "summary": scores["summary"],
                }
            )
    finally:
        client.close()

    n = len(score_matrix)
    if n == 0:
        raise SystemExit("No queries ran")

    aggregate: dict[str, Any] = {}
    for j, key in enumerate(AGG_KEYS):
        col = [score_matrix[i][j] for i in range(n)]
        aggregate[key] = round(sum(col) / len(col), 1)
    crit_means = [aggregate[k] for k in AGG_KEYS]
    aggregate["mean"] = round(sum(crit_means) / len(crit_means), 1) if crit_means else 0.0

    try:
        rel_golden = str(golden_path.relative_to(PROJECT_ROOT))
    except ValueError:
        rel_golden = str(golden_path)

    report = {
        "run_type": "golden_full",
        "iteration": iteration,
        "timestamp": ts,
        "golden_queries_path": rel_golden,
        "golden_queries_version": golden_version,
        "query_count": n,
        "synthesis_model": SYNTH_MODEL,
        "judge_model": JUDGE_MODEL,
        "aggregate": aggregate,
        "queries": queries_out,
    }

    out_path = out_dir / f"golden_iteration_{iteration}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Wrote %s", out_path)

    _print_stdout(iteration, ts, golden_rows, score_matrix, aggregate, out_path)
    return out_path


def _print_stdout(
    iteration: int,
    ts: str,
    golden_rows: list[dict[str, Any]],
    score_matrix: list[list[float]],
    aggregate: dict[str, Any],
    out_path: Path,
) -> None:
    col_w = 28
    print()
    print(f"Golden iteration {iteration} — {ts[:10]} ({len(golden_rows)} queries)")
    print(
        f"{'ID / Query':<{col_w}}  {'Adhere':>6}  {'Cite':>6}  {'Format':>6}  "
        f"{'Tone':>6}  {'Divers':>6}  {'Mean':>6}"
    )
    for row, scores_row in zip(golden_rows, score_matrix):
        gid = row["id"] or "?"
        label = query_label_short(row["query"], max_len=max(8, col_w - len(gid) - 2))
        label = f"{gid} {label}"
        if len(label) > col_w:
            label = label[: col_w - 1] + "…"
        if any(x > 0 for x in scores_row):
            qm = round(sum(scores_row) / len(scores_row), 1)
            print(
                f"{label:<{col_w}}  {int(scores_row[0]):>6}  {int(scores_row[1]):>6}  "
                f"{int(scores_row[2]):>6}  {int(scores_row[3]):>6}  {int(scores_row[4]):>6}  {qm:>6}"
            )
        else:
            print(
                f"{label:<{col_w}}  {'fail':>6}  {'fail':>6}  {'fail':>6}  "
                f"{'fail':>6}  {'fail':>6}  {'—':>6}"
            )

    mean_label = "MEAN (across queries)"
    print(
        f"{mean_label:<{col_w}}  {aggregate['instruction_adherence']:>6}  "
        f"{aggregate['citation_accuracy']:>6}  {aggregate['format_consistency']:>6}  "
        f"{aggregate['tone_intent']:>6}  {aggregate['diversity']:>6}  {aggregate['mean']:>6}"
    )
    print(f"\nFull report: {out_path}")


def main() -> None:
    root = PROJECT_ROOT
    default_golden = root / "data" / "golden_queries.json"
    ap = argparse.ArgumentParser(
        description="Run full golden_queries.json through retrieve → synthesis → judge.",
    )
    ap.add_argument(
        "--golden",
        type=Path,
        default=default_golden,
        help=f"Path to golden_queries.json (default: {default_golden})",
    )
    args = ap.parse_args()
    golden_path = args.golden.resolve()
    if not golden_path.is_file():
        raise SystemExit(f"Golden file not found: {golden_path}")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    _run_golden_iteration(golden_path=golden_path)


if __name__ == "__main__":
    main()
