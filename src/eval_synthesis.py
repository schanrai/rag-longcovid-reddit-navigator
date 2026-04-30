#!/usr/bin/env python3
"""
eval_synthesis.py — Phase 4 Step 0a: run fixed golden queries through retrieve →
synthesis, score each answer with an LLM judge (one JSON response per query).

For multi-model comparison (same retrieval, fixed judge), use:
    python -m src.eval_synthesis_bakeoff

Usage:
    cd projects/rag-longcovid-reddit-navigator
    python -m src.eval_synthesis

Writes: reports/synthesis_eval/iteration_<N>.json (N auto-increments).
Requires: OPENROUTER_API_KEY, VOYAGE_API_KEY, Weaviate env (same as pipeline_cli).
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.eval_synthesis_common import (
    AGG_KEYS,
    EVAL_QUERIES,
    JUDGE_MODEL,
    PROJECT_ROOT,
    SYNTH_MODEL,
    call_judge,
    default_judge_result,
    judge_user_payload,
    next_iteration_index,
    normalize_scores,
    query_label_short,
    synthesis_telemetry,
)
from src.retrieval.config import RetrievalConfig
from src.retrieval.hybrid_search import _build_weaviate_client
from src.retrieval.pipeline import retrieve
from src.synthesis import SynthesisConfig, generate_synthesis, pack_context

log = logging.getLogger("eval_synthesis")


def _run_single_model_iteration() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    voyage_key = os.environ.get("VOYAGE_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    if not voyage_key:
        raise SystemExit("VOYAGE_API_KEY is not set")

    syn_base = SynthesisConfig(model=SYNTH_MODEL)
    retrieval_cfg = RetrievalConfig()
    retrieval_cfg.reranker.enabled = True
    max_chunks = syn_base.max_chunks_in_context

    out_dir = PROJECT_ROOT / "reports" / "synthesis_eval"
    iteration = next_iteration_index(out_dir)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    queries_out: list[dict[str, Any]] = []
    score_matrix: list[list[float]] = []

    client = _build_weaviate_client()
    try:
        for i, query in enumerate(EVAL_QUERIES, start=1):
            log.info("Query %d/%d: %s", i, len(EVAL_QUERIES), query[:60])
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
                log.exception("Judge failed for query: %s", query[:50])
                scores = default_judge_result(str(exc))

            row_scores = [float(scores[k]) for k in AGG_KEYS]
            score_matrix.append(row_scores)

            q_mean = round(sum(row_scores) / len(row_scores), 1) if row_scores else 0.0
            queries_out.append(
                {
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

    report = {
        "run_type": "single",
        "iteration": iteration,
        "timestamp": ts,
        "synthesis_model": SYNTH_MODEL,
        "judge_model": JUDGE_MODEL,
        "aggregate": aggregate,
        "queries": queries_out,
    }

    out_path = out_dir / f"iteration_{iteration}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Wrote %s", out_path)

    _print_single_stdout(iteration, ts, score_matrix, aggregate, out_path)


def _print_single_stdout(
    iteration: int,
    ts: str,
    score_matrix: list[list[float]],
    aggregate: dict[str, Any],
    out_path: Path,
) -> None:
    col_w = 30
    print()
    print(f"Iteration {iteration} — {ts[:10]}")
    print(
        f"{'Query':<{col_w}}  {'Adhere':>6}  {'Cite':>6}  {'Format':>6}  {'Tone':>6}  {'Divers':>6}  {'Mean':>6}"
    )
    for row, q in zip(score_matrix, EVAL_QUERIES):
        label = query_label_short(q, max_len=col_w)
        if any(x > 0 for x in row):
            qm = round(sum(row) / len(row), 1)
            print(
                f"{label:<{col_w}}  {int(row[0]):>6}  {int(row[1]):>6}  {int(row[2]):>6}  "
                f"{int(row[3]):>6}  {int(row[4]):>6}  {qm:>6}"
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    _run_single_model_iteration()


if __name__ == "__main__":
    main()
