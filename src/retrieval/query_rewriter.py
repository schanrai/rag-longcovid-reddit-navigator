#!/usr/bin/env python3
"""
query_rewriter.py — Interactive LLM-based query rewrite + intent classification.

Two-mode design (see scope doc Section 5.5):
  - confident:     Single rewrite returned; retrieval proceeds immediately.
  - clarification: 2–3 candidate rewrites returned; frontend presents options.
                   Until frontend exists, best_rewrite is used as a fallback.

Intent categories (symptom / treatment / timeline / prevalence /
                   emotional / admin / community / meta / unknown)
are passed downstream to ranking.py for intent-driven freshness weighting.

Usage (standalone QA):
  cd projects/rag-longcovid-reddit-navigator
  python3 -m src.retrieval.query_rewriter --test
  python3 -m src.retrieval.query_rewriter --query "why am I so tired 8 months later"
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Final

import httpx
from dotenv import load_dotenv

# Allow running as a module or directly
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
load_dotenv(Path(__file__).parent.parent.parent / ".env")

from src.retrieval.config import RetrievalConfig, RewriterConfig
from src.retrieval.models import (
    IntentCategory,
    RewriteCandidate,
    RewriteMode,
    RewriteResult,
)

log = logging.getLogger("retrieval.query_rewriter")

# ── Prompts ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT: Final[str] = """You are a query preprocessing assistant for a Long COVID patient community RAG system.

Your job is to:
1. Rewrite informal or ambiguous health queries into clear, retrievable search strings.
2. Classify the query's intent category.
3. Assess how confident you are that you understand the user's intent.

Rules for rewriting:
- Expand medical abbreviations and informal shorthand to their full form, then stop — do not paraphrase or substitute further. Common examples: LC → Long COVID, BF → brain fog, PEM → post-exertional malaise, POTS → postural orthostatic tachycardia syndrome, MCAS → mast cell activation syndrome, LDN → low-dose naltrexone. Expand any other abbreviations you recognise from the Long COVID or chronic illness community — this list is not exhaustive. For example: "LDN" becomes "low-dose naltrexone", not "naltrexone therapy" or "opioid antagonist treatment".
- Remove social scaffolding that adds no retrieval value: Reddit-style openers ("Hi everyone", "long time lurker"), hedging preambles ("sorry if this has been asked"), and sign-offs ("thanks in advance"). Do NOT remove question framing ("why am I", "I want to know", "can anyone tell me") — these carry intent and should be preserved or rephrased into the query.
- IMPORTANT: Community experience phrases like "anyone?", "has anyone tried", "has anyone else experienced this?" are NOT scaffolding. They signal the user wants peer accounts and personal experiences. You MUST preserve them in the rewritten query — rephrase if needed but do not drop them. Example: "beta blockers — anyone?" should keep the community-seeking element, e.g. "beta blockers for tachycardia — community experiences".
- Preserve emotional tone signals — do not strip distress or urgency
- If the query is ambiguous, produce 2–3 distinct interpretations instead of one

Intent categories (pick the single best fit):
- symptom: questions about specific symptoms, their nature, or experience
- treatment: medications, therapies, interventions, management strategies, evidence questions (e.g. "does Paxlovid work")
- timeline: recovery timelines, progression, when things improve or worsen
- prevalence: how common symptoms/experiences are in the community
- emotional: coping, mental health, social support, validation, grief
- admin: insurance, disability benefits, workplace accommodations, sick leave, appeals — the logistics of managing a chronic illness
- community: subreddit navigation, community resources, general Long COVID information
- unknown: cannot determine intent

Respond ONLY with valid JSON matching this schema exactly:
{
  "mode": "confident" | "clarification",
  "intent": "<category>",
  "rewrites": [
    {
      "query": "<rewritten query string>",
      "explanation": "<why this rewrite>",
      "confidence": <float 0.0–1.0>
    }
  ]
}

For confident mode: one rewrite object with confidence ≥ 0.75.
For clarification mode: 2–3 rewrite objects, each confidence < 0.75, representing distinct interpretations.
Do not include any text outside the JSON object."""

# ── Core rewrite function ──────────────────────────────────────────────────────

def rewrite(
    query: str,
    *,
    cfg: RetrievalConfig | None = None,
    api_key: str | None = None,
) -> RewriteResult:
    """
    Rewrite a user query and classify its intent.

    Args:
        query:   Raw user query string.
        cfg:     RetrievalConfig (uses defaults if not provided).
        api_key: OpenRouter API key (falls back to OPENROUTER_API_KEY env var).

    Returns:
        RewriteResult with mode, intent, and rewrite candidates.

    Raises:
        ValueError: If the API key is missing or the LLM response is malformed.
        httpx.HTTPError: On network or API failures.
    """
    cfg = cfg or RetrievalConfig()
    rcfg: RewriterConfig = cfg.rewriter

    key = api_key or os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise ValueError("OPENROUTER_API_KEY is not set")

    payload: dict[str, Any] = {
        "model": rcfg.model,
        "temperature": rcfg.temperature,
        "max_tokens": rcfg.max_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
    }

    url = f"{rcfg.openrouter_base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.environ.get("APP_URL", "http://localhost:3000"),
    }

    log.debug("Calling OpenRouter: model=%s query=%r", rcfg.model, query)

    with httpx.Client(timeout=rcfg.request_timeout_s) as client:
        r = client.post(url, headers=headers, json=payload)
        r.raise_for_status()

    body = r.json()
    raw = body["choices"][0]["message"]["content"].strip()
    log.debug("Raw LLM response: %s", raw)

    return _parse_response(query, raw, cfg)


def _parse_response(
    original_query: str,
    raw: str,
    cfg: RetrievalConfig,
) -> RewriteResult:
    """Parse the LLM JSON response into a RewriteResult."""
    # Strip markdown code fences if the model wraps in ```json ... ```
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned non-JSON: {exc}\nRaw: {raw[:300]}") from exc

    # Validate and coerce intent
    try:
        intent = IntentCategory(data.get("intent", "unknown"))
    except ValueError:
        intent = IntentCategory.UNKNOWN
        log.warning("Unknown intent category %r — defaulting to 'unknown'", data.get("intent"))

    # Parse rewrites
    raw_rewrites = data.get("rewrites", [])
    if not raw_rewrites:
        raise ValueError(f"LLM returned no rewrites for query: {original_query!r}")

    candidates = [
        RewriteCandidate(
            query=r["query"],
            explanation=r.get("explanation", ""),
            confidence=float(r.get("confidence", 0.5)),
        )
        for r in raw_rewrites
    ]

    # Determine mode: use what the LLM returned, but validate against threshold
    llm_mode = data.get("mode", "confident")
    best_confidence = max(c.confidence for c in candidates)

    if llm_mode == "clarification" or best_confidence < cfg.rewriter.ambiguity_threshold:
        mode = RewriteMode.CLARIFICATION
    else:
        mode = RewriteMode.CONFIDENT
        # In confident mode keep only the best candidate
        candidates = [max(candidates, key=lambda c: c.confidence)]

    result = RewriteResult(
        mode=mode,
        original_query=original_query,
        rewrites=candidates,
        intent=intent,
        raw_llm_response=raw,
    )

    log.info(
        "Rewrite: mode=%s intent=%s best=%r confidence=%.2f",
        result.mode.value,
        result.intent.value,
        result.best_rewrite.query,
        result.best_rewrite.confidence,
    )

    return result


# ── Test suite (--test mode) ───────────────────────────────────────────────────

TEST_QUERIES: Final[list[dict[str, str]]] = [
    # ── Clean baseline cases ───────────────────────────────────────────────────
    {
        "query": "why am I still so tired 8 months later",
        "hint": "symptom — no abbreviations, informal phrasing",
    },
    {
        "query": "can I get disability benefits for LC",
        "hint": "admin — LC should expand; disability/insurance intent",
    },
    {
        "query": "LC",
        "hint": "clarification expected — single abbreviation, entirely ambiguous",
    },
    {
        "query": "Hi everyone, sorry if this has been asked before but I've been really struggling with fatigue and PEM for months, any advice? Thanks in advance",
        "hint": "treatment — social scaffolding (opener + apology + sign-off) should be stripped; PEM expanded; 'any advice?' is treatment-seeking, not generic filler",
    },

    # ── From golden queries — informal register + real ambiguity ──────────────
    {
        "query": "I feel like I've gotten dumber since my infection",
        "hint": "symptom — no medical terms at all; rewriter must infer brain fog / cognitive symptoms (q01)",
    },
    {
        "query": "People do recover, five years in and still making progress....my 2 cents",
        "hint": "timeline — a statement, not a question; tests non-question intent classification (q06)",
    },
    {
        "query": "I miss the old me",
        "hint": "emotional — 5 words, no medical terms; tests emotional intent from minimal input (q09)",
    },
    {
        "query": "Disability just got denied again and the final answer is NO",
        "hint": "admin — frustrated statement, not a query; tests intent from vented frustration (q10)",
    },
    {
        "query": "beta blockers for the constant tachycardia thing — anyone? ivabradine?",
        "hint": "treatment — multiple informal abbreviations; no full medical terms (q15)",
    },
    {
        "query": "Doctors suck! I have no respect for them nowadays",
        "hint": "emotional — angry vent; no medical terms; tone preservation critical (q17)",
    },
    {
        "query": "how do you even know if its PEM vs youre just out of shape",
        "hint": "symptom or community — genuinely ambiguous; clarification mode possible (q18)",
    },
    {
        "query": "LC awareness. Doctor and family don't believe me =(",
        "hint": "emotional — abbreviated LC + dual disbelief; the q19 eval failure case (q19)",
    },
]


COMPARE_MODELS: Final[list[str]] = [
    "google/gemini-2.5-flash-lite",   # default — fast, cheap
    "google/gemini-2.5-flash",        # step up — same family, more capable
    "openai/gpt-4o-mini",             # cross-architecture validation
]


def run_tests(cfg: RetrievalConfig) -> tuple[int, int]:
    """Run the built-in QA test suite and print results."""
    print("\n" + "=" * 70)
    print(f"query_rewriter.py — QA test suite  [{cfg.rewriter.model}]")
    print("=" * 70)

    passed = 0
    failed = 0

    for i, case in enumerate(TEST_QUERIES, start=1):
        q = case["query"]
        hint = case["hint"]
        print(f"\n[{i:02d}] Query: {q!r}")
        print(f"      Hint:  {hint}")
        try:
            result = rewrite(q, cfg=cfg)
            print(f"      Mode:    {result.mode.value}")
            print(f"      Intent:  {result.intent.value}")
            for j, c in enumerate(result.rewrites, start=1):
                print(f"      Rewrite {j}: {c.query!r}  (conf={c.confidence:.2f})")
                print(f"               → {c.explanation}")
            passed += 1
        except Exception as exc:
            print(f"      ERROR: {exc}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed / {len(TEST_QUERIES)} total")
    print("=" * 70)
    return passed, failed


def run_compare(models: list[str]) -> None:
    """
    Run all test queries across multiple models and print a side-by-side comparison.

    For each query, shows intent + mode + best rewrite for every model so differences
    are immediately visible. Errors are flagged inline rather than stopping the run.

    Focus areas for review:
      - Intent accuracy on ambiguous queries (q01, q09, q18)
      - Clarification trigger on "LC" alone and q18
      - Abbreviation expansion without over-paraphrasing (q15, q19)
      - Tone preservation on emotional queries (q09, q17)
      - Social scaffolding stripping (long opener query)
    """
    col_w = 38  # width per model column

    header = "  ".join(m.split("/")[-1][:col_w].ljust(col_w) for m in models)
    print("\n" + "=" * (col_w * len(models) + 2 * (len(models) - 1)))
    print(f"query_rewriter.py — model comparison  ({len(models)} models, {len(TEST_QUERIES)} queries)")
    print("=" * (col_w * len(models) + 2 * (len(models) - 1)))
    print(header)

    totals: dict[str, dict[str, int]] = {m: {"passed": 0, "failed": 0} for m in models}

    for i, case in enumerate(TEST_QUERIES, start=1):
        q = case["query"]
        hint = case["hint"]
        print(f"\n[{i:02d}] {q!r}")
        print(f"      Hint: {hint}")
        print("      " + "-" * (col_w * len(models) + 2 * (len(models) - 1)))

        results: list[str] = []
        for model in models:
            cfg = RetrievalConfig()
            cfg.rewriter.model = model
            try:
                r = rewrite(q, cfg=cfg)
                best = r.best_rewrite
                cell = (
                    f"mode={r.mode.value}  intent={r.intent.value}\n"
                    f"      rewrite: {best.query[:col_w - 16]!r}\n"
                    f"      conf={best.confidence:.2f}"
                )
                totals[model]["passed"] += 1
            except Exception as exc:
                cell = f"ERROR: {str(exc)[:col_w - 8]}"
                totals[model]["failed"] += 1
            results.append(cell)

        # Print row — one line per cell line, columns aligned
        cell_lines = [c.split("\n") for c in results]
        max_lines = max(len(cl) for cl in cell_lines)
        for line_i in range(max_lines):
            row_parts = []
            for cl in cell_lines:
                part = cl[line_i] if line_i < len(cl) else ""
                row_parts.append(part.ljust(col_w))
            print("      " + "  ".join(row_parts))

    print("\n" + "=" * (col_w * len(models) + 2 * (len(models) - 1)))
    print("Summary:")
    for model in models:
        t = totals[model]
        print(f"  {model.split('/')[-1]:<35} passed={t['passed']}  failed={t['failed']}")
    print("=" * (col_w * len(models) + 2 * (len(models) - 1)))


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    ap = argparse.ArgumentParser(description="Query rewriter — interactive two-mode design")
    ap.add_argument("--query", "-q", type=str, default=None,
                    help="Single query to rewrite (prints result as JSON)")
    ap.add_argument("--test", action="store_true",
                    help="Run built-in QA test suite with the default (or --model) model")
    ap.add_argument("--compare", action="store_true",
                    help=f"Run all test queries across {len(COMPARE_MODELS)} models side-by-side")
    ap.add_argument("--models", type=str, default=None,
                    help="Comma-separated model IDs for --compare (overrides default set)")
    ap.add_argument("--model", type=str, default=None,
                    help="Override OpenRouter model for --test or --query")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Enable DEBUG logging")
    args = ap.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    cfg = RetrievalConfig()
    if args.model:
        cfg.rewriter.model = args.model

    if args.compare:
        models = (
            [m.strip() for m in args.models.split(",")]
            if args.models
            else COMPARE_MODELS
        )
        run_compare(models)
    elif args.test:
        run_tests(cfg)
    elif args.query:
        result = rewrite(args.query, cfg=cfg)
        print(result.model_dump_json(indent=2, exclude={"raw_llm_response"}))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
