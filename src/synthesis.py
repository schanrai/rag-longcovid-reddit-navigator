#!/usr/bin/env python3
"""
synthesis.py — Grounded answer synthesis with citations.

Builds a Gemini-optimized prompt from retrieval output, calls the LLM via
OpenRouter, and returns a structured response payload for the API/frontend.

Interim telemetry (pre–Module 6): configure the `src.synthesis` logger (or root)
for your environment. At DEBUG: full system prompt, full user prompt, and raw
LLM message body before JSON parse. At INFO: cited-vs-provided chunk ratio per
request. No external tracing; defer Arize/Phoenix to Module 6.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

from src.retrieval.models import RetrievalResult, SearchResult

load_dotenv()

log = logging.getLogger(__name__)

DISCLAIMER_TEXT = "This tool surfaces community experience, not medical advice."
ANCHOR_PATTERN = re.compile(r"\[(\d+)\]")
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
SYSTEM_PROMPT_PATH = PROMPTS_DIR / "synthesis_system_prompt.txt"
USER_PROMPT_PATH = PROMPTS_DIR / "synthesis_user_prompt.txt"

# When the model returns prose or unrecoverable JSON, re-call OpenRouter (same messages)
# before failing — keeps eval runs from aborting on occasional format slips.
_SYNTHESIS_PARSE_MAX_ATTEMPTS = 3


class Source(BaseModel):
    anchor: int = Field(..., ge=1, description="Citation anchor number used in answer markdown.")
    chunk_id: str = Field(..., description="Chunk identifier used for this citation.")
    permalink: str = Field(default="", description="Source permalink for frontend citation cards.")
    post_title: str = Field(default="", description="Parent thread title.")
    chunk_type: str = Field(default="", description="Chunk type: comment or post.")
    comment_score: int | None = Field(default=None, description="Comment upvote score, when source is a comment.")
    post_score: int | None = Field(default=None, description="Post upvote score, when source is a post.")
    num_comments: int | None = Field(default=None, description="Number of comments in the post, when source is a post.")
    created_utc: int | None = Field(default=None, description="Unix timestamp from source metadata.")


class ResponseMetadata(BaseModel):
    model: str
    temperature: float
    latency_ms: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    chunks_provided: int


class SynthesisResponse(BaseModel):
    answer: str = Field(..., description="Markdown synthesis with inline [n] anchors.")
    sources: list[Source] = Field(default_factory=list)
    disclaimer: str = DISCLAIMER_TEXT
    metadata: ResponseMetadata


class SynthesisConfig(BaseModel):
    model: str = "google/gemini-3-flash-preview"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    temperature: float = 0.1
    max_tokens: int = 1800
    timeout_s: float = 60.0
    max_chunks_in_context: int = 25
    retry_attempts: int = Field(
        default=4,
        ge=1,
        description="Total HTTP attempts (first try + retries) for transient OpenRouter failures.",
    )
    retry_base_delay_s: float = Field(
        default=1.0,
        ge=0.0,
        description="Base delay for exponential backoff between retries (seconds).",
    )
    retry_max_delay_s: float = Field(
        default=60.0,
        ge=0.0,
        description="Upper cap on sleep between retries and on Retry-After (seconds).",
    )
    answer_word_target_min: int = Field(
        default=200,
        ge=50,
        le=800,
        description=(
            "Soft minimum word count for the entire answer_markdown string (editorial guidance): "
            "headings, bullet prose, anchors like [1], and the disclaimer all count toward this band. "
            "Pydantic ge/le: inclusive allowed range for this numeric setting."
        ),
    )
    answer_word_target_max: int = Field(
        default=520,
        ge=80,
        le=1200,
        description=(
            "Soft maximum word count for the entire answer_markdown string (same scope as min). "
            "Pydantic ge/le: inclusive allowed range for this numeric setting."
        ),
    )
    topic_heading_target_min: int = Field(
        default=2,
        ge=1,
        le=12,
        description=(
            "Minimum distinct **Topic Heading** sections when sources support them; "
            "pair with answer_word_* for short, one-paragraph-per-topic answers."
        ),
    )
    topic_heading_target_max: int = Field(
        default=4,
        ge=1,
        le=12,
        description=(
            "Maximum distinct **Topic Heading** sections; keep low when word budget is tight "
            "(e.g. one short paragraph of prose per heading within answer_word_*)."
        ),
    )

    @model_validator(mode="after")
    def _validate_editorial_targets(self) -> SynthesisConfig:
        if self.answer_word_target_min > self.answer_word_target_max:
            raise ValueError(
                "answer_word_target_min must be <= answer_word_target_max "
                f"({self.answer_word_target_min} > {self.answer_word_target_max})"
            )
        if self.topic_heading_target_min > self.topic_heading_target_max:
            raise ValueError(
                "topic_heading_target_min must be <= topic_heading_target_max "
                f"({self.topic_heading_target_min} > {self.topic_heading_target_max})"
            )
        return self


# HTTP status codes worth retrying (rate limits + gateway / upstream instability).
_RETRYABLE_HTTP_STATUSES: frozenset[int] = frozenset({408, 429, 502, 503, 504})


def _retry_after_seconds(headers: httpx.Headers) -> float | None:
    """Parse Retry-After as delay seconds; None if absent or not a plain integer/float."""
    raw = headers.get("Retry-After")
    if not raw:
        return None
    raw = raw.strip()
    try:
        return float(raw)
    except ValueError:
        # HTTP-date form — skip; caller falls back to exponential backoff.
        return None


def _http_status_retryable(status_code: int) -> bool:
    return status_code in _RETRYABLE_HTTP_STATUSES


def _backoff_seconds(
    *,
    attempt: int,
    base_delay_s: float,
    max_delay_s: float,
    retry_after_s: float | None,
) -> float:
    """Compute sleep before the next attempt (attempt is 0-based index of the failed try)."""
    if retry_after_s is not None and retry_after_s >= 0:
        wait = retry_after_s
    else:
        exp = base_delay_s * (2**attempt)
        jitter = 0.5 + random.random()
        wait = exp * jitter
    return min(max_delay_s, max(0.0, wait))


def _post_openrouter_chat_completions(
    *,
    client: httpx.Client,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    cfg: SynthesisConfig,
) -> httpx.Response:
    """
    POST chat/completions with retries for transient failures.

    Distinguishes timeout exhaustion (raises TimeoutError) from other failures
    (re-raises the last exception).
    """
    last_exc: BaseException | None = None
    last_was_timeout = False

    for attempt in range(cfg.retry_attempts):
        try:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response
        except httpx.TimeoutException as exc:
            last_exc = exc
            last_was_timeout = True
            log.warning(
                "OpenRouter synthesis timeout (attempt %s/%s, timeout_s=%s): %s",
                attempt + 1,
                cfg.retry_attempts,
                cfg.timeout_s,
                exc,
            )
        except httpx.HTTPStatusError as exc:
            last_exc = exc
            last_was_timeout = False
            code = exc.response.status_code
            if not _http_status_retryable(code):
                raise
            retry_after: float | None = None
            if code == 429:
                retry_after = _retry_after_seconds(exc.response.headers)
            log.warning(
                "OpenRouter synthesis HTTP %s (attempt %s/%s)%s: %s",
                code,
                attempt + 1,
                cfg.retry_attempts,
                f", Retry-After={retry_after}s" if retry_after is not None else "",
                exc,
            )
        except httpx.RequestError as exc:
            last_exc = exc
            last_was_timeout = False
            log.warning(
                "OpenRouter synthesis transport error (attempt %s/%s): %s",
                attempt + 1,
                cfg.retry_attempts,
                exc,
            )

        if attempt + 1 >= cfg.retry_attempts:
            break

        retry_after_sleep: float | None = None
        if isinstance(last_exc, httpx.HTTPStatusError) and last_exc.response.status_code == 429:
            retry_after_sleep = _retry_after_seconds(last_exc.response.headers)

        sleep_s = _backoff_seconds(
            attempt=attempt,
            base_delay_s=cfg.retry_base_delay_s,
            max_delay_s=cfg.retry_max_delay_s,
            retry_after_s=retry_after_sleep,
        )
        if sleep_s > 0:
            time.sleep(sleep_s)

    assert last_exc is not None
    if last_was_timeout:
        raise TimeoutError(
            f"OpenRouter synthesis request timed out after {cfg.retry_attempts} attempt(s) "
            f"(timeout_s={cfg.timeout_s})."
        ) from last_exc
    raise last_exc


def _format_chunk_block(anchor: int, result: SearchResult) -> str:
    m = result.metadata
    score = m.comment_score if m.comment_score is not None else m.post_score
    body = (result.text or "").strip()
    return (
        f"<SOURCE_{anchor}>\n"
        f"chunk_id: {result.chunk_id}\n"
        f"chunk_type: {m.chunk_type}\n"
        f"post_title: {m.post_title}\n"
        f"permalink: {m.permalink}\n"
        f"created_utc: {m.created_utc}\n"
        f"comment_score: {score}\n"
        f"num_comments: {m.num_comments}\n"
        f"text: {body}\n"
        f"</SOURCE_{anchor}>"
    )


def pack_context(results: list[SearchResult], *, max_chunks: int) -> str:
    """
    Pack retrieval results into structured, delimited context blocks.

    Anchors map 1:1 to position in this packed context.
    """
    blocks: list[str] = []
    for idx, item in enumerate(results[:max_chunks], start=1):
        blocks.append(_format_chunk_block(idx, item))
    return "\n\n".join(blocks)


def _load_prompt_template(path: Path) -> str:
    if not path.exists():
        raise ValueError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def build_system_prompt(cfg: SynthesisConfig) -> str:
    """Load and render the system prompt template (includes editorial length / topic targets)."""
    template = _load_prompt_template(SYSTEM_PROMPT_PATH)
    return template.format(
        disclaimer_text=DISCLAIMER_TEXT,
        answer_word_target_min=cfg.answer_word_target_min,
        answer_word_target_max=cfg.answer_word_target_max,
        topic_heading_target_min=cfg.topic_heading_target_min,
        topic_heading_target_max=cfg.topic_heading_target_max,
    )


def build_user_prompt(retrieval: RetrievalResult, packed_context: str) -> str:
    """Load and render the user prompt template."""
    template = _load_prompt_template(USER_PROMPT_PATH)
    rewritten_query = retrieval.query.best_rewrite.query
    return template.format(
        original_query=retrieval.query.original_query,
        rewritten_query=rewritten_query,
        packed_context=packed_context,
    )


def _strip_json_fences(raw: str) -> str:
    if not raw.startswith("```"):
        return raw.strip()
    lines = raw.splitlines()
    cleaned = [line for line in lines if not line.strip().startswith("```")]
    return "\n".join(cleaned).strip()


def _extract_answer_markdown_loose(cleaned: str) -> str | None:
    """
    Recover answer_markdown when the model emits invalid JSON (common case:
    unescaped \" inside the answer_markdown string).

    Assumes a single top-level \"answer_markdown\" string (optional trailing keys
    after a comma are handled by treating a quote followed by , or } as closing).
    """
    match = re.search(r'"answer_markdown"\s*:\s*"', cleaned)
    if not match:
        return None
    i = match.end()
    out: list[str] = []
    n = len(cleaned)
    while i < n:
        c = cleaned[i]
        if c == "\\" and i + 1 < n:
            esc = cleaned[i + 1]
            if esc == "n":
                out.append("\n")
                i += 2
                continue
            if esc == "t":
                out.append("\t")
                i += 2
                continue
            if esc == "r":
                out.append("\r")
                i += 2
                continue
            if esc == '"':
                out.append('"')
                i += 2
                continue
            if esc == "\\":
                out.append("\\")
                i += 2
                continue
            if esc == "u" and i + 6 <= n:
                hex_part = cleaned[i + 2 : i + 6]
                if len(hex_part) == 4 and all(
                    ch in "0123456789abcdefABCDEF" for ch in hex_part
                ):
                    out.append(chr(int(hex_part, 16)))
                    i += 6
                    continue
            out.append(c)
            out.append(esc)
            i += 2
            continue
        if c == '"':
            tail = cleaned[i + 1 :].lstrip()
            if tail.startswith("}") or tail.startswith(","):
                break
            out.append('"')
            i += 1
            continue
        out.append(c)
        i += 1
    text = "".join(out)
    return text.strip() or None


def _parse_synthesis_payload(cleaned: str) -> dict[str, Any]:
    """Parse strict JSON, or fall back to loose extraction of answer_markdown."""
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        loose = _extract_answer_markdown_loose(cleaned)
        if loose is None:
            raise ValueError(
                f"Synthesis model returned non-JSON output: {cleaned[:300]}"
            ) from exc
        log.warning(
            "synthesis JSON parse failed (%s); recovered answer_markdown via loose extractor",
            exc,
        )
        return {"answer_markdown": loose}


def _extract_sources(answer_markdown: str, results: list[SearchResult]) -> list[Source]:
    anchors_in_text = {int(match) for match in ANCHOR_PATTERN.findall(answer_markdown)}
    valid_anchors = sorted(a for a in anchors_in_text if 1 <= a <= len(results))
    sources: list[Source] = []
    for anchor in valid_anchors:
        result = results[anchor - 1]
        m = result.metadata
        sources.append(
            Source(
                anchor=anchor,
                chunk_id=result.chunk_id,
                permalink=m.permalink or "",
                post_title=m.post_title or "",
                chunk_type=m.chunk_type or "",
                comment_score=m.comment_score,
                post_score=m.post_score,
                num_comments=m.num_comments,
                created_utc=m.created_utc,
            )
        )
    return sources


def generate_synthesis(
    retrieval: RetrievalResult,
    *,
    cfg: SynthesisConfig | None = None,
    api_key: str | None = None,
) -> SynthesisResponse:
    """
    Generate grounded synthesis from retrieval output.

    Args:
        retrieval: Retrieval pipeline output (query + ranked results).
        cfg: Optional synthesis config overrides.
        api_key: Optional OpenRouter key; falls back to OPENROUTER_API_KEY.

    Raises:
        ValueError: missing key, empty context, invalid LLM output.
        TimeoutError: all attempts failed due to request timeout (see chained cause).
        httpx.HTTPError: non-retryable HTTP errors or exhausted retries.
        httpx.RequestError: exhausted retries on transport errors.

    On invalid or empty ``answer_markdown`` JSON envelope, the OpenRouter call is
    retried up to ``_SYNTHESIS_PARSE_MAX_ATTEMPTS`` times (same prompt) before raising.

    Logging:
        DEBUG — full system prompt, full user prompt, raw assistant message before JSON parse.
        INFO — cited N of M chunks, plus model id and wall latency (synthesis HTTP only).
    """
    cfg = cfg or SynthesisConfig()
    key = (api_key or os.environ.get("OPENROUTER_API_KEY", "")).strip()
    if not key:
        raise ValueError("OPENROUTER_API_KEY is not set")
    if not retrieval.results:
        raise ValueError("Retrieval results are empty; cannot synthesize grounded response.")

    context_results = retrieval.results[: cfg.max_chunks_in_context]
    packed_context = pack_context(
        context_results,
        max_chunks=cfg.max_chunks_in_context,
    )
    system_prompt = build_system_prompt(cfg)
    user_prompt = build_user_prompt(retrieval, packed_context)
    log.debug("synthesis system prompt (full):\n%s", system_prompt)
    log.debug("synthesis user prompt (full):\n%s", user_prompt)

    payload: dict[str, Any] = {
        "model": cfg.model,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.environ.get("APP_URL", "http://localhost:3000"),
    }
    url = f"{cfg.openrouter_base_url.rstrip('/')}/chat/completions"

    started = time.perf_counter()
    body: dict[str, Any] = {}
    answer_markdown = ""
    with httpx.Client(timeout=cfg.timeout_s) as client:
        for parse_attempt in range(_SYNTHESIS_PARSE_MAX_ATTEMPTS):
            response = _post_openrouter_chat_completions(
                client=client,
                url=url,
                headers=headers,
                payload=payload,
                cfg=cfg,
            )
            body = response.json()
            raw = body["choices"][0]["message"]["content"].strip()
            log.debug("synthesis raw LLM response (pre-parse, full):\n%s", raw)
            cleaned = _strip_json_fences(raw)
            try:
                parsed = _parse_synthesis_payload(cleaned)
                answer_markdown = (parsed.get("answer_markdown") or "").strip()
                if not answer_markdown:
                    raise ValueError("Synthesis model returned empty answer_markdown.")
            except ValueError as exc:
                log.warning(
                    "synthesis JSON parse failed (attempt %d/%d): %s",
                    parse_attempt + 1,
                    _SYNTHESIS_PARSE_MAX_ATTEMPTS,
                    exc,
                )
                if parse_attempt + 1 >= _SYNTHESIS_PARSE_MAX_ATTEMPTS:
                    raise
                time.sleep(min(2.0, 0.5 * (2**parse_attempt)))
                continue
            break
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    sources = _extract_sources(answer_markdown, context_results)
    provided = len(context_results)
    cited = len(sources)
    log.info(
        "synthesis citation coverage: cited %d of %d chunks (model=%s, latency_ms=%d)",
        cited,
        provided,
        cfg.model,
        elapsed_ms,
    )
    usage = body.get("usage", {})
    metadata = ResponseMetadata(
        model=cfg.model,
        temperature=cfg.temperature,
        latency_ms=elapsed_ms,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        total_tokens=usage.get("total_tokens"),
        chunks_provided=len(context_results),
    )

    return SynthesisResponse(
        answer=answer_markdown,
        sources=sources,
        disclaimer=DISCLAIMER_TEXT,
        metadata=metadata,
    )
