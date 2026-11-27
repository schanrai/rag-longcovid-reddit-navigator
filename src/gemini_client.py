"""
One Gemini generateContent call, shared by the rewriter, synthesis, and citation verifier.

Reads GEMINI_API_KEY from the environment. Does not log the key or the request headers.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Literal

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.openrouter_retry import post_openrouter_chat_completions

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=True)

log = logging.getLogger(__name__)

_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
_SUCCESS_FINISH_REASONS: frozenset[str] = frozenset({"STOP", "FINISH_REASON_UNSPECIFIED"})

ThinkingLevel = Literal["minimal", "low", "medium", "high"]


class GeminiResponseError(Exception):
    """The HTTP call succeeded and the body cannot be used as answer text."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Gemini response unusable: {reason}")


class GeminiCallConfig(BaseModel):
    """Retry and generation settings for one generateContent call."""

    temperature: float = 0.0
    max_output_tokens: int = Field(
        ...,
        ge=1,
        description="Caller passes the existing max_tokens for that call: 512 rewriter, 1800 synthesis, 4096 verifier.",
    )
    timeout_s: float = Field(default=60.0, gt=0)
    retry_attempts: int = Field(
        default=4,
        ge=1,
        description="Total HTTP attempts (first try + retries) for timeouts, 429, and gateway errors.",
    )
    retry_base_delay_s: float = Field(default=1.0, ge=0)
    retry_max_delay_s: float = Field(default=60.0, ge=0)
    base_url: str = _GEMINI_BASE_URL


def extract_response_text(body: dict[str, Any]) -> str:
    """
    Join non-thought text parts from the first candidate.

    Raises GeminiResponseError when the prompt was blocked, no candidate exists,
    or finishReason is anything other than a normal stop (including MAX_TOKENS
    and SAFETY). Partial text on those finishes is discarded so callers do not
    parse a truncated JSON envelope.
    """
    feedback = body.get("promptFeedback") or {}
    block_reason = feedback.get("blockReason")
    if block_reason:
        raise GeminiResponseError(f"blockReason={block_reason}")

    candidates = body.get("candidates")
    if not candidates:
        raise GeminiResponseError("candidates missing")

    candidate = candidates[0]
    finish = candidate.get("finishReason")
    if finish and finish not in _SUCCESS_FINISH_REASONS:
        raise GeminiResponseError(f"finishReason={finish}")

    parts = (candidate.get("content") or {}).get("parts") or []
    texts = [
        part["text"]
        for part in parts
        if isinstance(part, dict) and part.get("text") and not part.get("thought")
    ]
    if not texts:
        raise GeminiResponseError(f"no text parts finishReason={finish}")
    return "".join(texts).strip()


def generate_text(
    *,
    model: str,
    user_prompt: str,
    max_output_tokens: int,
    system_prompt: str = "",
    api_key: str | None = None,
    temperature: float = 0.0,
    timeout_s: float = 60.0,
    retry_attempts: int = 4,
    retry_base_delay_s: float = 1.0,
    retry_max_delay_s: float = 60.0,
    thinking_level: ThinkingLevel | None = None,
) -> str:
    """
    Call Gemini generateContent and return the answer text.

    Args:
        model: Gemini model id, without a vendor prefix (for example ``gemini-3-flash-preview``).
        user_prompt: The user turn.
        system_prompt: Optional system instruction. Omitted from the body when blank.
        max_output_tokens: Existing cap for this call. Rewriter 512, synthesis 1800, verifier 4096.
            Thinking tokens count against this cap on Gemini 3 models.
        api_key: Overrides ``GEMINI_API_KEY``.
        thinking_level: Sent as ``thinkingConfig.thinkingLevel`` when set. When ``None``,
            no thinking config is sent and the model uses its own default.

    Raises:
        ValueError: Missing key, empty model, or empty user prompt.
        GeminiResponseError: The body is blocked, truncated, or has no answer text.
        TimeoutError: Every attempt timed out.
        httpx.HTTPError: Non-retryable HTTP status, or retries exhausted.
        httpx.RequestError: Retries exhausted on a transport error.
    """
    cfg = GeminiCallConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        timeout_s=timeout_s,
        retry_attempts=retry_attempts,
        retry_base_delay_s=retry_base_delay_s,
        retry_max_delay_s=retry_max_delay_s,
    )
    key = (api_key or os.environ.get("GEMINI_API_KEY", "")).strip()
    if not key:
        raise ValueError("GEMINI_API_KEY is not set")
    model_id = model.strip()
    if not model_id:
        raise ValueError("Gemini model id is empty")
    if not user_prompt.strip():
        raise ValueError("user_prompt is empty")

    generation_config: dict[str, Any] = {
        "temperature": cfg.temperature,
        "maxOutputTokens": cfg.max_output_tokens,
    }
    if thinking_level is not None:
        generation_config["thinkingConfig"] = {"thinkingLevel": thinking_level}
    payload: dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": generation_config,
    }
    if system_prompt.strip():
        payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

    url = f"{cfg.base_url.rstrip('/')}/models/{model_id}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": key,
    }
    log.info(
        "gemini generate_text model=%s max_output_tokens=%s thinking_level=%s",
        model_id,
        cfg.max_output_tokens,
        thinking_level,
    )
    with httpx.Client(timeout=cfg.timeout_s) as client:
        response = post_openrouter_chat_completions(
            client=client,
            url=url,
            headers=headers,
            payload=payload,
            cfg=cfg,
            log_label="Gemini",
        )
    return extract_response_text(response.json())
