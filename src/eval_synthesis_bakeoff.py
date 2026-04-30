#!/usr/bin/env python3
"""
eval_synthesis_bakeoff.py — Phase 4 Step 0b: model bake-off.

Same five golden queries, one retrieve() per query (cached in memory), then each
candidate synthesis model generates an answer from the same packed context; one
fixed judge scores every answer on the same five dimensions.

Usage:
    cd projects/rag-longcovid-reddit-navigator
    python -m src.eval_synthesis_bakeoff

Writes: reports/synthesis_eval/bakeoff_<N>.json
Requires: OPENROUTER_API_KEY, VOYAGE_API_KEY, Weaviate env (same as pipeline_cli).
OpenRouter model IDs: verify on https://openrouter.ai if a provider renames a slug.
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
    aggregate_from_rows,
    call_judge,
    default_judge_result,
    judge_user_payload,
    next_bakeoff_index,
    normalize_scores,
    synthesis_telemetry,
)
from src.retrieval.config import RetrievalConfig
from src.retrieval.hybrid_search import _build_weaviate_client
from src.retrieval.models import RetrievalResult
from src.retrieval.pipeline import retrieve
from src.synthesis import SynthesisConfig, generate_synthesis, pack_context

log = logging.getLogger("eval_synthesis_bakeoff")

# Display label + OpenRouter id (https://openrouter.ai)
BAKEOFF_CANDIDATES: list[dict[str, str]] = [
    {"id": "openai/gpt-oss-120b", "label": "GPT OSS 120B"},
    {"id": "google/gemini-2.5-flash", "label": "Gemini 2.5 Flash"},
    {"id": "google/gemini-3-flash-preview", "label": "Gemini 3 Flash (preview)"},
    {"id": "anthropic/claude-haiku-4.5", "label": "Claude Haiku 4.5"},
    {"id": "openai/gpt-5.4-mini", "label": "GPT-5.4-mini"},
]


def _run_bakeoff() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    voyage_key = os.environ.get("VOYAGE_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    if not voyage_key:
        raise SystemExit("VOYAGE_API_KEY is not set")

    syn_base = SynthesisConfig()
    retrieval_cfg = RetrievalConfig()
    retrieval_cfg.reranker.enabled = True
    max_chunks = syn_base.max_chunks_in_context
    model_ids = [c["id"] for c in BAKEOFF_CANDIDATES]

    out_dir = PROJECT_ROOT / "reports" / "synthesis_eval"
    bake_id = next_bakeoff_index(out_dir)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    retrievals: list[tuple[str, RetrievalResult]] = []
    client = _build_weaviate_client()
    try:
        for i, query in enumerate(EVAL_QUERIES, start=1):
            log.info("Bakeoff retrieve %d/%d: %s", i, len(EVAL_QUERIES), query[:60])
            r = retrieve(
                query,
                cfg=retrieval_cfg,
                voyage_api_key=voyage_key,
                weaviate_client=client,
            )
            retrievals.append((query, r))
    finally:
        client.close()

    per_model_score_rows: dict[str, list[dict[str, float]]] = {mid: [] for mid in model_ids}
    queries_out: list[dict[str, Any]] = []

    for query, retrieval in retrievals:
        ctx_results = retrieval.results[:max_chunks]
        packed = pack_context(ctx_results, max_chunks=max_chunks)
        original = retrieval.query.original_query
        rewritten = retrieval.query.best_rewrite.query

        by_model: dict[str, Any] = {}

        for c in BAKEOFF_CANDIDATES:
            mid = c["id"]
            label = c["label"]
            cfg = syn_base.model_copy(update={"model": mid})
            try:
                syn_resp = generate_synthesis(retrieval, cfg=cfg)
            except Exception as exc:
                log.exception("Synthesis failed model=%s query=%s", mid, query[:40])
                per_model_score_rows[mid].append({k: 0.0 for k in AGG_KEYS})
                by_model[mid] = {
                    "label": label,
                    "error": str(exc),
                    "answer_markdown": "",
                    "sources_cited": 0,
                    "synthesis": None,
                    "scores": {k: 0 for k in AGG_KEYS} | {"mean": 0.0},
                    "issues": [f"synthesis_error: {exc}"],
                    "summary": f"Synthesis failed: {exc}",
                }
                continue

            judge_user = judge_user_payload(
                original_query=original,
                rewritten_query=rewritten,
                answer_markdown=syn_resp.answer,
                packed_context=packed,
            )
            try:
                judge_raw = call_judge(judge_user, api_key=api_key)
                scores = normalize_scores(judge_raw)
            except Exception as exc:
                log.exception("Judge failed model=%s query=%s", mid, query[:40])
                scores = default_judge_result(str(exc))

            row = {k: float(scores[k]) for k in AGG_KEYS}
            per_model_score_rows[mid].append(row)

            q_mean = round(sum(row.values()) / len(row), 1) if row else 0.0
            by_model[mid] = {
                "label": label,
                "error": None,
                "answer_markdown": syn_resp.answer,
                "sources_cited": len(syn_resp.sources),
                "synthesis": synthesis_telemetry(syn_resp),
                "scores": {**{k: scores[k] for k in AGG_KEYS}, "mean": q_mean},
                "issues": scores["issues"],
                "summary": scores["summary"],
            }

        queries_out.append(
            {
                "query": query,
                "rewritten_query": rewritten,
                "sources_provided": len(ctx_results),
                "by_model": by_model,
            }
        )

    aggregate_by_model: dict[str, Any] = {}
    for mid in model_ids:
        label = next(c["label"] for c in BAKEOFF_CANDIDATES if c["id"] == mid)
        rows = per_model_score_rows.get(mid, [])
        aggregate_by_model[mid] = {
            "label": label,
            **aggregate_from_rows(rows),
        }

    report = {
        "run_type": "bakeoff",
        "bakeoff": bake_id,
        "timestamp": ts,
        "judge_model": JUDGE_MODEL,
        "candidates": BAKEOFF_CANDIDATES,
        "aggregate_by_model": aggregate_by_model,
        "queries": queries_out,
    }

    out_path = out_dir / f"bakeoff_{bake_id}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Wrote %s", out_path)

    _print_bakeoff_stdout(bake_id, ts, aggregate_by_model, model_ids, out_path)


def _print_bakeoff_stdout(
    bake_id: int,
    ts: str,
    aggregate_by_model: dict[str, Any],
    model_ids: list[str],
    out_path: Path,
) -> None:
    print()
    print(f"Bakeoff {bake_id} — {ts[:10]}  |  judge: {JUDGE_MODEL}")
    col_w = 22
    hdr = f"{'Model':<{col_w}}  {'Adh':>5}  {'Cite':>5}  {'Fmt':>5}  {'Tone':>5}  {'Div':>5}  {'Mean':>5}"
    print(hdr)
    print("-" * len(hdr))
    for mid in model_ids:
        a = aggregate_by_model[mid]
        label = a.get("label", mid)
        short = label[: col_w - 1] + "…" if len(label) > col_w else label
        if a.get("instruction_adherence", 0) or a.get("citation_accuracy", 0):
            print(
                f"{short:<{col_w}}  {a['instruction_adherence']:>5.1f}  {a['citation_accuracy']:>5.1f}  "
                f"{a['format_consistency']:>5.1f}  {a['tone_intent']:>5.1f}  {a['diversity']:>5.1f}  {a['mean']:>5.1f}"
            )
        else:
            print(f"{short:<{col_w}}  {'(no valid scores)':>35}")
    print(f"\nFull report: {out_path}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    _run_bakeoff()


if __name__ == "__main__":
    main()
