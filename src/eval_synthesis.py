#!/usr/bin/env python3
"""
eval_synthesis.py — Phase 4 Step 0a: run fixed golden queries through retrieve →
synthesis, score each answer with an LLM judge (one JSON response per query).

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
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.retrieval.config import RetrievalConfig
from src.retrieval.hybrid_search import _build_weaviate_client
from src.retrieval.pipeline import retrieve
from src.synthesis import SynthesisConfig, generate_synthesis, pack_context

log = logging.getLogger("eval_synthesis")

SYNTH_MODEL = "google/gemini-2.5-flash"
JUDGE_MODEL = "google/gemini-2.5-flash"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

EVAL_QUERIES: list[str] = [
    "Anyone taking beta blockers for tachycardia-like symptoms?",
    "I miss my life",
    "Has LDN helped anyone with Long COVID symptoms?",
    "why am i still so exhausted 8 months after getting Covid?",
    "Finally recovered after 4 years - here are the treatments I tried",
]

JUDGE_SYSTEM = """You are a strict evaluator for a Long COVID community RAG synthesis.

You receive the user's original query, the rewritten query, intent, the model's answer markdown (with [n] citation anchors), and the exact source chunks shown to the synthesis model (<SOURCE_1> … </SOURCE_1>, etc.).

Your job:
1. For EVERY [n] anchor appearing in the answer markdown, verify that the claim(s) in the same sentence or immediately preceding clause are directly supported by the text inside <SOURCE_n>. List EVERY mismatch and EVERY orphaned anchor (anchor with no defensible support in SOURCE_n). Do not sample anchors — check all of them.
2. Score five criteria from 1 (poor) to 5 (excellent):
   - instruction_adherence: Does every factual claim come from the provided chunks only? Penalize outside knowledge or invention.
   - citation_accuracy: Consistency of each [n] with SOURCE_n after your full anchor audit.
   - format_consistency: Bold topic headings, point paragraphs with citations at end of points — not bullet lists or rigid generic subsections unless clearly appropriate.
   - tone_intent: Appropriate for intent (e.g. clinical-neutral for treatment queries, empathetic for emotional queries).
   - diversity: Multiple distinct angles when sources support them; incompatible community views called out when sources conflict.

Output ONLY a single valid JSON object (no markdown fences, no commentary). Shape:
{
  "instruction_adherence": <int 1-5>,
  "citation_accuracy": <int 1-5>,
  "format_consistency": <int 1-5>,
  "tone_intent": <int 1-5>,
  "diversity": <int 1-5>,
  "issues": ["<string>", ...],
  "summary": "<2-3 sentences: what worked, what failed, the single most important fix>"
}

issues must be a JSON array (use [] if none). summary is required."""


def _strip_json_fences(raw: str) -> str:
    text = raw.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    cleaned = [ln for ln in lines if not ln.strip().startswith("```")]
    return "\n".join(cleaned).strip()


def _next_iteration(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    best = 0
    for p in out_dir.glob("iteration_*.json"):
        m = re.match(r"iteration_(\d+)\.json$", p.name)
        if m:
            best = max(best, int(m.group(1)))
    return best + 1


def _query_label(q: str, max_len: int = 30) -> str:
    q = q.replace("\n", " ").strip()
    return (q[: max_len - 1] + "…") if len(q) > max_len else q


def _parse_judge_json(raw: str) -> dict[str, Any]:
    cleaned = _strip_json_fences(raw)
    return json.loads(cleaned)


def _judge_payload(
    *,
    original_query: str,
    rewritten_query: str,
    intent: str,
    answer_markdown: str,
    packed_context: str,
) -> str:
    return (
        f"<ORIGINAL_QUERY>\n{original_query}\n</ORIGINAL_QUERY>\n\n"
        f"<REWRITTEN_QUERY>\n{rewritten_query}\n</REWRITTEN_QUERY>\n\n"
        f"<INTENT>\n{intent}\n</INTENT>\n\n"
        f"<ANSWER_MARKDOWN>\n{answer_markdown}\n</ANSWER_MARKDOWN>\n\n"
        f"<PACKED_SOURCES>\n{packed_context}\n</PACKED_SOURCES>"
    )


def _call_judge(
    user_content: str,
    *,
    api_key: str,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.environ.get("APP_URL", "http://localhost:3000"),
    }
    payload: dict[str, Any] = {
        "model": JUDGE_MODEL,
        "temperature": 0.2,
        "max_tokens": 4096,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user_content},
        ],
    }
    with httpx.Client(timeout=timeout_s) as client:
        for attempt in range(2):
            r = client.post(OPENROUTER_URL, headers=headers, json=payload)
            if r.status_code == 429 and attempt == 0:
                ra = r.headers.get("Retry-After")
                try:
                    sleep_s = float(ra) if ra is not None else 5.0
                except (TypeError, ValueError):
                    sleep_s = 5.0
                log.warning("Judge 429; retrying after %.1fs", sleep_s)
                time.sleep(sleep_s)
                continue
            r.raise_for_status()
            body = r.json()
            raw = body["choices"][0]["message"]["content"].strip()
            return _parse_judge_json(raw)


def _default_judge_result(exc: str) -> dict[str, Any]:
    return {
        "instruction_adherence": 0,
        "citation_accuracy": 0,
        "format_consistency": 0,
        "tone_intent": 0,
        "diversity": 0,
        "issues": [f"judge_parse_or_http_error: {exc}"],
        "summary": "Judge call failed or returned invalid JSON; no reliable scores for this query.",
    }


def _normalize_scores(d: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "instruction_adherence",
        "citation_accuracy",
        "format_consistency",
        "tone_intent",
        "diversity",
    )
    out = dict(d)
    for k in keys:
        v = out.get(k)
        try:
            iv = int(round(float(v)))
        except (TypeError, ValueError):
            iv = 0
        out[k] = max(1, min(5, iv)) if 1 <= iv <= 5 else 0
    issues = out.get("issues")
    if not isinstance(issues, list):
        issues = []
    out["issues"] = [str(x) for x in issues]
    out["summary"] = str(out.get("summary") or "").strip() or "(no summary from judge)"
    return out


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    voyage_key = os.environ.get("VOYAGE_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    if not voyage_key:
        raise SystemExit("VOYAGE_API_KEY is not set")

    syn_cfg = SynthesisConfig(model=SYNTH_MODEL)
    retrieval_cfg = RetrievalConfig()
    retrieval_cfg.reranker.enabled = True
    max_chunks = syn_cfg.max_chunks_in_context

    out_dir = _PROJECT_ROOT / "reports" / "synthesis_eval"
    iteration = _next_iteration(out_dir)
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
            syn_resp = generate_synthesis(retrieval, cfg=syn_cfg)
            ctx_results = retrieval.results[:max_chunks]
            packed = pack_context(ctx_results, max_chunks=max_chunks)

            judge_user = _judge_payload(
                original_query=retrieval.query.original_query,
                rewritten_query=retrieval.query.best_rewrite.query,
                intent=retrieval.query.intent.value,
                answer_markdown=syn_resp.answer,
                packed_context=packed,
            )
            try:
                judge_raw = _call_judge(judge_user, api_key=api_key)
                scores = _normalize_scores(judge_raw)
            except Exception as exc:
                log.exception("Judge failed for query: %s", query[:50])
                scores = _default_judge_result(str(exc))

            row_scores = [
                float(scores["instruction_adherence"]),
                float(scores["citation_accuracy"]),
                float(scores["format_consistency"]),
                float(scores["tone_intent"]),
                float(scores["diversity"]),
            ]
            score_matrix.append(row_scores)

            q_mean = round(sum(row_scores) / len(row_scores), 1) if row_scores else 0.0
            queries_out.append(
                {
                    "query": query,
                    "rewritten_query": retrieval.query.best_rewrite.query,
                    "intent": retrieval.query.intent.value,
                    "answer_markdown": syn_resp.answer,
                    "sources_cited": len(syn_resp.sources),
                    "sources_provided": len(ctx_results),
                    "scores": {
                        "instruction_adherence": scores["instruction_adherence"],
                        "citation_accuracy": scores["citation_accuracy"],
                        "format_consistency": scores["format_consistency"],
                        "tone_intent": scores["tone_intent"],
                        "diversity": scores["diversity"],
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

    agg_keys = [
        "instruction_adherence",
        "citation_accuracy",
        "format_consistency",
        "tone_intent",
        "diversity",
    ]
    aggregate: dict[str, Any] = {}
    for j, key in enumerate(agg_keys):
        col = [score_matrix[i][j] for i in range(n)]
        aggregate[key] = round(sum(col) / len(col), 1)
    crit_means = [aggregate[k] for k in agg_keys]
    aggregate["mean"] = round(sum(crit_means) / len(crit_means), 1) if crit_means else 0.0

    report = {
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

    # Stdout summary table
    print()
    print(f"Iteration {iteration} — {ts[:10]}")
    col_w = 30
    print(
        f"{'Query':<{col_w}}  {'Adhere':>6}  {'Cite':>6}  {'Format':>6}  {'Tone':>6}  {'Divers':>6}  {'Mean':>6}"
    )
    for row, q in zip(score_matrix, EVAL_QUERIES):
        label = _query_label(q, max_len=col_w)
        if any(x > 0 for x in row):
            qm = round(sum(row) / len(row), 1)
            print(
                f"{label:<{col_w}}  {int(row[0]):>6}  {int(row[1]):>6}  {int(row[2]):>6}  "
                f"{int(row[3]):>6}  {int(row[4]):>6}  {qm:>6}"
            )
        else:
            print(f"{label:<{col_w}}  {'fail':>6}  {'fail':>6}  {'fail':>6}  {'fail':>6}  {'fail':>6}  {'—':>6}")

    mean_label = "MEAN (across queries)"
    print(
        f"{mean_label:<{col_w}}  {aggregate['instruction_adherence']:>6}  "
        f"{aggregate['citation_accuracy']:>6}  {aggregate['format_consistency']:>6}  "
        f"{aggregate['tone_intent']:>6}  {aggregate['diversity']:>6}  {aggregate['mean']:>6}"
    )
    print(f"\nFull report: {out_path}")


if __name__ == "__main__":
    main()
