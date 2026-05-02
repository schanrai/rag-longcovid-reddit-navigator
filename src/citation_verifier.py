"""
Post-synthesis citation audit: drop [n] anchors where SOURCE_n does not explicitly support the attached claim.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Protocol

import httpx

from src.openrouter_retry import SupportsOpenRouterRetry, post_openrouter_chat_completions

log = logging.getLogger(__name__)

VERIFIER_MODEL = "google/gemini-2.5-flash"
ANCHOR_PATTERN = re.compile(r"\[(\d+)\]")

_VERIFIER_SYSTEM_PROMPT = """You are a citation consistency auditor for community-health summaries.

For each citation anchor [n] that appears in ANSWER_MARKDOWN:
1. Locate the specific factual claim in the prose that [n] is attached to (usually immediately before or after the bracket token).
2. Find SOURCE_n inside PACKED_SOURCES (the block between <SOURCE_n> and </SOURCE_n>).
3. Set keep=true only if SOURCE_n directly and explicitly supports that exact claim — not merely the same broad topic, not by plausible inference alone.
4. If one sentence bundles multiple facts (e.g. helped with A, B, and C) but SOURCE_n clearly supports only some of them, set keep=false — do not let [n] stand for an over-broad bundled claim.

Output ONLY valid JSON with no markdown fences and no other text:
{"verdicts":[{"anchor":<int>,"keep":<bool>,"reason":"<brief string>"},...]}

Include exactly one verdict object per distinct anchor integer that appears in ANSWER_MARKDOWN. Do not emit verdicts for anchor numbers that do not appear in the answer."""


class _VerifierCfg(SupportsOpenRouterRetry, Protocol):
    openrouter_base_url: str


def _strip_json_fences(raw: str) -> str:
    if not raw.startswith("```"):
        return raw.strip()
    lines = raw.splitlines()
    cleaned = [line for line in lines if not line.strip().startswith("```")]
    return "\n".join(cleaned).strip()


def _anchors_in_markdown(answer_markdown: str) -> list[int]:
    found = {int(m) for m in ANCHOR_PATTERN.findall(answer_markdown)}
    return sorted(found)


def _strip_anchor_tokens(text: str, removed: list[int]) -> str:
    out = text
    for n in removed:
        out = re.sub(rf"\[{n}\]", "", out)
    return out


def verify_citations(
    answer_markdown: str,
    packed_context: str,
    *,
    api_key: str,
    cfg: _VerifierCfg,
) -> tuple[str, list[int], int]:
    """
    Audit each [n] in ``answer_markdown`` against ``packed_context`` and remove unsupported anchors.

    Returns:
        (cleaned_markdown, sorted removed anchor ints, verifier wall-clock latency in ms)
    """
    anchors = _anchors_in_markdown(answer_markdown)
    if not anchors:
        return answer_markdown, [], 0

    user_prompt = (
        "<ANSWER_MARKDOWN>\n"
        f"{answer_markdown}\n"
        "</ANSWER_MARKDOWN>\n\n"
        "<PACKED_SOURCES>\n"
        f"{packed_context}\n"
        "</PACKED_SOURCES>"
    )

    log.debug("citation verifier system prompt (full):\n%s", _VERIFIER_SYSTEM_PROMPT)
    log.debug("citation verifier user prompt (full):\n%s", user_prompt)

    payload: dict[str, Any] = {
        "model": VERIFIER_MODEL,
        "temperature": 0,
        "max_tokens": 4096,
        "messages": [
            {"role": "system", "content": _VERIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.environ.get("APP_URL", "http://localhost:3000"),
    }
    url = f"{cfg.openrouter_base_url.rstrip('/')}/chat/completions"

    started = time.perf_counter()
    try:
        with httpx.Client(timeout=cfg.timeout_s) as client:
            response = post_openrouter_chat_completions(
                client=client,
                url=url,
                headers=headers,
                payload=payload,
                cfg=cfg,
                log_label="OpenRouter citation_verifier",
            )
        body = response.json()
        raw = body["choices"][0]["message"]["content"].strip()
        log.debug("citation verifier raw LLM response (full):\n%s", raw)
        cleaned = _strip_json_fences(raw)
        data = json.loads(cleaned)
    except Exception as exc:
        log.warning("citation verifier failed; leaving anchors unchanged: %s", exc)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return answer_markdown, [], elapsed_ms

    elapsed_ms = int((time.perf_counter() - started) * 1000)

    verdicts_raw = data.get("verdicts")
    if not isinstance(verdicts_raw, list):
        log.warning("citation verifier: invalid verdicts shape; leaving anchors unchanged")
        return answer_markdown, [], elapsed_ms

    verdict_by_anchor: dict[int, dict[str, Any]] = {}
    for item in verdicts_raw:
        if not isinstance(item, dict):
            continue
        try:
            a = int(item["anchor"])
        except (KeyError, TypeError, ValueError):
            continue
        verdict_by_anchor[a] = item

    anchor_set = set(anchors)
    removed: list[int] = []
    for a in anchors:
        v = verdict_by_anchor.get(a)
        if v is None:
            log.warning("citation verifier: no verdict for anchor %s; keeping", a)
            continue
        if not bool(v.get("keep", True)):
            removed.append(a)

    removed = sorted(x for x in removed if x in anchor_set)

    kept_count = len(anchor_set) - len(removed)
    log.info(
        "citation verifier: kept %d of %d anchors (removed: %s)",
        kept_count,
        len(anchor_set),
        removed,
    )

    cleaned_md = _strip_anchor_tokens(answer_markdown, removed)
    return cleaned_md, removed, elapsed_ms
