"""
Shared helpers for synthesis evaluation scripts (single-model and bake-off).

Imported by eval_synthesis.py and eval_synthesis_bakeoff.py — not a CLI entry point.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.synthesis import SynthesisResponse

log = logging.getLogger(__name__)

SYNTH_MODEL = "google/gemini-3-flash-preview"
JUDGE_MODEL = os.environ.get("EVAL_JUDGE_MODEL", "google/gemini-2.5-flash").strip() or "google/gemini-2.5-flash"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

EVAL_QUERIES: list[str] = [
    "Anyone taking beta blockers for tachycardia-like symptoms?",
    "I miss my life",
    "Has LDN helped anyone with Long COVID symptoms?",
    "why am i still so exhausted 8 months after getting Covid?",
    "Finally recovered after 4 years - here are the treatments I tried",
]

JUDGE_SYSTEM = """You are a rigorous evaluator for a Long COVID community RAG synthesis.

You receive the user's original query, the rewritten query, the model's answer markdown (with [n] citation anchors), and the exact source chunks shown to the synthesis model (<SOURCE_1> … </SOURCE_1>, etc.). You do NOT receive a separate intent label — infer what the query is asking (e.g. treatment vs emotional vs mixed) from the wording of the queries only.

Your job:

1. instruction_adherence (for scoring): Does every factual claim come from the provided chunks only? Penalize outside knowledge or invention.

2. citation_accuracy — use ONLY these rules when scoring citation_accuracy and when deciding whether something is a citation issue (do not apply conjunctive per-anchor semantics):
   - Collective / distributive semantics: When multiple anchors appear together on the same sentence or clause (e.g. [1][14][15] at the end), evaluate them as a group. The cited sources taken together should cover the substantive claims in that sentence. Do NOT require each cited source, by itself, to support every clause or phrase in the sentence.
   - Flag an individual [n] only if SOURCE_n has no relevant support for any substantive part of the claim that citation group is attached to — not because it omits a detail that another cited source in the same group covers.
   - Example (equivalent on citation_accuracy): "Side effects include grogginess [1], cold hands and feet [14], and nightmares [15]" vs "Side effects include grogginess, cold hands and feet, and nightmares [1][14][15]" — same score when the three sources collectively cover those three effects. Do NOT flag [1] for not mentioning cold hands if [14] covers cold hands.
   - Reasonable paraphrasing: If meaning is preserved, do NOT treat synonym substitution or normal synthesis language as citation inaccuracies — including e.g. "life-changing" vs "changed my life"; "heart palpitations" vs "fluttery rapid heart"; "disappeared entirely" vs "went away"; "manageable baseline" as synthesis when the source describes shorter or less severe crashes (or similar); light interpretive glue (e.g. "focus energy on moving forward") when it aligns with the source's idea.
   - DO still flag citation_accuracy problems when: the meaning materially changes vs what sources support; a cited source contradicts the claim; a cited source has zero relevance to any part of the claim the group attaches to; or information is fabricated / absent from all provided sources.

3. Examine every [n] in the answer; list issues only for real violations of instruction_adherence or of the citation_accuracy rules above — not for acceptable collective coverage or reasonable paraphrase.

4. Score five criteria from 1 (poor) to 5 (excellent):
   - instruction_adherence: Does every factual claim come from the provided chunks only? Penalize outside knowledge or invention.
   - citation_accuracy: Anchor usage judged only by the citation_accuracy rules in section 2 (collective coverage + paraphrase tolerance).
   - format_consistency: Bold topic headings, point paragraphs with citations at end of points — not bullet lists or rigid generic subsections unless clearly appropriate.
   - tone_intent: Is tone appropriate for what the queries ask and for patient-community context (e.g. clinical-neutral vs empathetic), judged from the query text alone — not from any external intent tag.
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

AGG_KEYS = [
    "instruction_adherence",
    "citation_accuracy",
    "format_consistency",
    "tone_intent",
    "diversity",
]


def _strip_json_fences(raw: str) -> str:
    text = raw.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    cleaned = [ln for ln in lines if not ln.strip().startswith("```")]
    return "\n".join(cleaned).strip()


def next_iteration_index(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    best = 0
    for p in out_dir.glob("iteration_*.json"):
        m = re.match(r"iteration_(\d+)\.json$", p.name)
        if m:
            best = max(best, int(m.group(1)))
    return best + 1


def next_bakeoff_index(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    best = 0
    for p in out_dir.glob("bakeoff_*.json"):
        m = re.match(r"bakeoff_(\d+)\.json$", p.name)
        if m:
            best = max(best, int(m.group(1)))
    return best + 1


def query_label_short(q: str, max_len: int = 30) -> str:
    q = q.replace("\n", " ").strip()
    return (q[: max_len - 1] + "…") if len(q) > max_len else q


def parse_judge_json(raw: str) -> dict[str, Any]:
    cleaned = _strip_json_fences(raw)
    return json.loads(cleaned)


def judge_user_payload(
    *,
    original_query: str,
    rewritten_query: str,
    answer_markdown: str,
    packed_context: str,
) -> str:
    return (
        f"<ORIGINAL_QUERY>\n{original_query}\n</ORIGINAL_QUERY>\n\n"
        f"<REWRITTEN_QUERY>\n{rewritten_query}\n</REWRITTEN_QUERY>\n\n"
        f"<ANSWER_MARKDOWN>\n{answer_markdown}\n</ANSWER_MARKDOWN>\n\n"
        f"<PACKED_SOURCES>\n{packed_context}\n</PACKED_SOURCES>"
    )


def call_judge(
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
        "temperature": 0.0,
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
            return parse_judge_json(raw)
    raise RuntimeError("judge: exhausted retries without return")


def default_judge_result(exc: str) -> dict[str, Any]:
    return {
        "instruction_adherence": 0,
        "citation_accuracy": 0,
        "format_consistency": 0,
        "tone_intent": 0,
        "diversity": 0,
        "issues": [f"judge_parse_or_http_error: {exc}"],
        "summary": "Judge call failed or returned invalid JSON; no reliable scores for this query.",
    }


def normalize_scores(d: dict[str, Any]) -> dict[str, Any]:
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


def synthesis_telemetry(m: SynthesisResponse) -> dict[str, Any]:
    md = m.metadata
    return {
        "model": md.model,
        "temperature": md.temperature,
        "latency_ms": md.latency_ms,
        "prompt_tokens": md.prompt_tokens,
        "completion_tokens": md.completion_tokens,
        "total_tokens": md.total_tokens,
        "chunks_provided": md.chunks_provided,
    }


def aggregate_from_rows(rows: list[dict[str, float]]) -> dict[str, Any]:
    if not rows:
        return {k: 0.0 for k in AGG_KEYS} | {"mean": 0.0}
    out: dict[str, Any] = {}
    for k in AGG_KEYS:
        col = [r[k] for r in rows if 1 <= r.get(k, 0) <= 5]
        out[k] = round(sum(col) / len(col), 1) if col else 0.0
    crit = [out[k] for k in AGG_KEYS]
    out["mean"] = round(sum(crit) / len(crit), 1) if crit else 0.0
    return out
