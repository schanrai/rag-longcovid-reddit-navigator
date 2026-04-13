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
        "hint": "symptom — social scaffolding (opener + apology + sign-off) should be stripped; PEM expanded; question preserved",
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


def run_tests(cfg: RetrievalConfig) -> None:
    """Run the built-in QA test suite and print results."""
    print("\n" + "=" * 70)
    print("query_rewriter.py — QA test suite")
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
                    help="Run built-in QA test suite across 10 test queries")
    ap.add_argument("--model", type=str, default=None,
                    help="Override OpenRouter model (e.g. openai/gpt-4o-mini)")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Enable DEBUG logging")
    args = ap.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    cfg = RetrievalConfig()
    if args.model:
        cfg.rewriter.model = args.model

    if args.test:
        run_tests(cfg)
    elif args.query:
        result = rewrite(args.query, cfg=cfg)
        print(result.model_dump_json(indent=2, exclude={"raw_llm_response"}))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
