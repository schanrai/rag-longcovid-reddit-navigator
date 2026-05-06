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
    python -m src.eval_synthesis_golden --only q21-q28

Writes: reports/synthesis_eval/golden_iteration_<N>.json (N auto-increments; separate from
iteration_<N>.json from eval_synthesis.py).

Dual-judge calibration (same retrieval + synthesis; second LLM scores the same payload):

    python -m src.eval_synthesis_golden --second-judge-model openai/gpt-5.4-mini

Primary judge: ``EVAL_JUDGE_MODEL`` (default ``anthropic/claude-sonnet-4.6``).

Compare two reports (e.g. baseline vs new run): ``python -m src.compare_golden_iterations --latest 2``.

Requires: OPENROUTER_API_KEY, VOYAGE_API_KEY, Weaviate env (same as pipeline_cli).
"""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

# Load project .env before any src imports (override=True: project keys win over shell).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=True)

import argparse
import json
import logging
import os
import re
from datetime import datetime, timezone
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

_RANGE_RE = re.compile(r"^q(\d+)-q(\d+)$", re.IGNORECASE)
_SINGLE_RE = re.compile(r"^q(\d+)$", re.IGNORECASE)


def _normalize_golden_id(num: int) -> str:
    """Match typical golden_queries.json ids (q01, q02, …)."""
    if 1 <= num <= 99:
        return f"q{num:02d}"
    return f"q{num}"


def parse_only_filter(spec: str) -> set[str]:
    """
    Parse --only value: comma-separated ids and/or inclusive ranges.

    Examples: ``q21-q28``, ``q01,q05-q08``, ``q3`` (normalized to ``q03``).
    """
    spec = spec.strip()
    if not spec:
        raise ValueError("--only must be non-empty when provided")
    out: set[str] = set()
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        m_range = _RANGE_RE.match(part)
        if m_range:
            lo = int(m_range.group(1))
            hi = int(m_range.group(2))
            a, b = min(lo, hi), max(lo, hi)
            for n in range(a, b + 1):
                out.add(_normalize_golden_id(n))
            continue
        m_single = _SINGLE_RE.match(part)
        if m_single:
            out.add(_normalize_golden_id(int(m_single.group(1))))
            continue
        raise ValueError(
            f"Invalid --only fragment {part!r}; use qNN or qNN-qMM (comma-separated)."
        )
    if not out:
        raise ValueError("--only produced no ids")
    return out


def filter_golden_rows(
    rows: list[dict[str, Any]],
    *,
    only_ids: set[str] | None,
) -> tuple[list[dict[str, Any]], set[str]]:
    """Preserve file order; return (filtered_rows, missing_requested_ids)."""
    if only_ids is None:
        return rows, set()
    available = {r["id"] for r in rows}
    missing = only_ids - available
    filtered = [r for r in rows if r["id"] in only_ids]
    return filtered, missing


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


def _run_golden_iteration(
    *,
    golden_path: Path,
    only_spec: str | None,
    second_judge_model: str | None,
) -> Path:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    voyage_key = os.environ.get("VOYAGE_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    if not voyage_key:
        raise SystemExit("VOYAGE_API_KEY is not set")

    all_rows, golden_version = load_golden_query_rows(golden_path)
    only_ids: set[str] | None = None
    if only_spec is not None:
        only_ids = parse_only_filter(only_spec)
    golden_rows, missing_ids = filter_golden_rows(all_rows, only_ids=only_ids)
    if missing_ids:
        log.warning(
            "Requested id(s) not found in golden file (skipped): %s",
            ", ".join(sorted(missing_ids)),
        )
    if not golden_rows:
        raise SystemExit(
            "No queries match --only filter (check ids exist in golden_queries.json)."
        )
    if only_spec:
        log.info(
            "Subset run: %d query(s) after --only %r",
            len(golden_rows),
            only_spec,
        )

    syn_base = SynthesisConfig(model=SYNTH_MODEL)
    retrieval_cfg = RetrievalConfig()
    retrieval_cfg.reranker.enabled = True
    max_chunks = syn_base.max_chunks_in_context

    out_dir = PROJECT_ROOT / "reports" / "synthesis_eval"
    iteration = next_golden_iteration_index(out_dir)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    queries_out: list[dict[str, Any]] = []
    score_matrix: list[list[float]] = []
    score_matrix_b: list[list[float]] | None = [] if second_judge_model else None

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
            q_payload: dict[str, Any] = {
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

            if second_judge_model:
                assert score_matrix_b is not None
                try:
                    raw_b = call_judge(
                        judge_user,
                        api_key=api_key,
                        model=second_judge_model,
                    )
                    scores_b = normalize_scores(raw_b)
                except Exception as exc:
                    log.exception(
                        "Secondary judge failed for golden_id=%s model=%s",
                        row["id"],
                        second_judge_model,
                    )
                    scores_b = default_judge_result(str(exc))
                row_b = [float(scores_b[k]) for k in AGG_KEYS]
                score_matrix_b.append(row_b)
                qb_mean = round(sum(row_b) / len(row_b), 1) if row_b else 0.0
                q_payload["scores_secondary"] = {
                    **{k: scores_b[k] for k in AGG_KEYS},
                    "mean": qb_mean,
                }
                q_payload["issues_secondary"] = scores_b["issues"]
                q_payload["summary_secondary"] = scores_b["summary"]

            queries_out.append(q_payload)
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

    aggregate_secondary: dict[str, Any] | None = None
    aggregate_delta: dict[str, Any] | None = None
    if second_judge_model and score_matrix_b is not None and len(score_matrix_b) == n:
        aggregate_secondary = {}
        for j, key in enumerate(AGG_KEYS):
            col_b = [score_matrix_b[i][j] for i in range(n)]
            aggregate_secondary[key] = round(sum(col_b) / len(col_b), 1)
        crit_b = [aggregate_secondary[k] for k in AGG_KEYS]
        aggregate_secondary["mean"] = (
            round(sum(crit_b) / len(crit_b), 1) if crit_b else 0.0
        )
        aggregate_delta = {
            key: round(aggregate_secondary[key] - aggregate[key], 2) for key in AGG_KEYS
        }
        aggregate_delta["mean"] = round(
            aggregate_secondary["mean"] - aggregate["mean"], 2
        )

    try:
        rel_golden = str(golden_path.relative_to(PROJECT_ROOT))
    except ValueError:
        rel_golden = str(golden_path)

    report = {
        "run_type": "golden_subset" if only_spec else "golden_full",
        "iteration": iteration,
        "timestamp": ts,
        "golden_queries_path": rel_golden,
        "golden_queries_version": golden_version,
        "only_filter": only_spec,
        "query_count": n,
        "synthesis_model": SYNTH_MODEL,
        "judge_model": JUDGE_MODEL,
        "judge_model_secondary": second_judge_model,
        "aggregate": aggregate,
        "aggregate_secondary": aggregate_secondary,
        "aggregate_delta_secondary_minus_primary": aggregate_delta,
        "queries": queries_out,
    }

    out_path = out_dir / f"golden_iteration_{iteration}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Wrote %s", out_path)

    _print_stdout(
        iteration,
        ts,
        golden_rows,
        score_matrix,
        aggregate,
        out_path,
        score_matrix_secondary=score_matrix_b,
        aggregate_secondary=aggregate_secondary,
        aggregate_delta=aggregate_delta,
        judge_secondary_label=second_judge_model,
    )
    return out_path


def _print_stdout(
    iteration: int,
    ts: str,
    golden_rows: list[dict[str, Any]],
    score_matrix: list[list[float]],
    aggregate: dict[str, Any],
    out_path: Path,
    *,
    score_matrix_secondary: list[list[float]] | None = None,
    aggregate_secondary: dict[str, Any] | None = None,
    aggregate_delta: dict[str, Any] | None = None,
    judge_secondary_label: str | None = None,
) -> None:
    col_w = 28
    print()
    dual = score_matrix_secondary is not None and judge_secondary_label
    title_extra = f" | dual judge (+ {judge_secondary_label})" if dual else ""
    print(
        f"Golden iteration {iteration} — {ts[:10]} ({len(golden_rows)} queries){title_extra}"
    )
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

    mean_label = f"MEAN primary ({JUDGE_MODEL.split('/')[-1][:18]})"
    print(
        f"{mean_label:<{col_w}}  {aggregate['instruction_adherence']:>6}  "
        f"{aggregate['citation_accuracy']:>6}  {aggregate['format_consistency']:>6}  "
        f"{aggregate['tone_intent']:>6}  {aggregate['diversity']:>6}  {aggregate['mean']:>6}"
    )

    if dual and score_matrix_secondary and aggregate_secondary:
        short_b = (judge_secondary_label or "").split("/")[-1][:18]
        print()
        print(f"Secondary judge ({short_b}) — same answers & sources")
        print(
            f"{'ID / Query':<{col_w}}  {'Adhere':>6}  {'Cite':>6}  {'Format':>6}  "
            f"{'Tone':>6}  {'Divers':>6}  {'Mean':>6}"
        )
        for row, scores_row in zip(golden_rows, score_matrix_secondary):
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
        mean_label_b = f"MEAN secondary ({short_b})"
        print(
            f"{mean_label_b:<{col_w}}  {aggregate_secondary['instruction_adherence']:>6}  "
            f"{aggregate_secondary['citation_accuracy']:>6}  {aggregate_secondary['format_consistency']:>6}  "
            f"{aggregate_secondary['tone_intent']:>6}  {aggregate_secondary['diversity']:>6}  "
            f"{aggregate_secondary['mean']:>6}"
        )

    if aggregate_delta:
        print()
        print("Δ (secondary − primary) aggregate:")
        print(
            f"{'':<{col_w}}  {aggregate_delta['instruction_adherence']:>6}  "
            f"{aggregate_delta['citation_accuracy']:>6}  {aggregate_delta['format_consistency']:>6}  "
            f"{aggregate_delta['tone_intent']:>6}  {aggregate_delta['diversity']:>6}  "
            f"{aggregate_delta['mean']:>6}"
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
    ap.add_argument(
        "--only",
        type=str,
        default=None,
        metavar="SPEC",
        help=(
            "Run only these golden ids (comma-separated). Each token is qNN or inclusive "
            "range qNN-qMM, e.g. q21-q28 or q01,q05-q08. Ids are normalized (q5 → q05)."
        ),
    )
    ap.add_argument(
        "--second-judge-model",
        type=str,
        default=None,
        metavar="OPENROUTER_MODEL",
        help=(
            "Optional second judge (OpenRouter id). Same rubric and user payload as primary; "
            "written to scores_secondary and aggregate_secondary for calibration. "
            "If omitted, uses EVAL_JUDGE_MODEL_SECOND when set."
        ),
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
    sj = (
        (args.second_judge_model or "").strip()
        or os.environ.get("EVAL_JUDGE_MODEL_SECOND", "").strip()
        or None
    )
    _run_golden_iteration(
        golden_path=golden_path,
        only_spec=args.only,
        second_judge_model=sj,
    )


if __name__ == "__main__":
    main()
