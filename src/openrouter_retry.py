"""
Shared OpenRouter chat/completions POST with retries (timeouts, 429, gateway errors).
"""
from __future__ import annotations

import logging
import random
import time
from typing import Any, Protocol

import httpx

log = logging.getLogger(__name__)

_RETRYABLE_HTTP_STATUSES: frozenset[int] = frozenset({408, 429, 502, 503, 504})


class SupportsOpenRouterRetry(Protocol):
    """Structural type for config objects that carry retry / timeout settings."""

    retry_attempts: int
    retry_base_delay_s: float
    retry_max_delay_s: float
    timeout_s: float


def _retry_after_seconds(headers: httpx.Headers) -> float | None:
    raw = headers.get("Retry-After")
    if not raw:
        return None
    raw = raw.strip()
    try:
        return float(raw)
    except ValueError:
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
    if retry_after_s is not None and retry_after_s >= 0:
        wait = retry_after_s
    else:
        exp = base_delay_s * (2**attempt)
        jitter = 0.5 + random.random()
        wait = exp * jitter
    return min(max_delay_s, max(0.0, wait))


def post_openrouter_chat_completions(
    *,
    client: httpx.Client,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    cfg: SupportsOpenRouterRetry,
    log_label: str = "OpenRouter",
) -> httpx.Response:
    """
    POST chat/completions with retries for transient failures.

    ``log_label`` prefixes warning messages (e.g. ``OpenRouter synthesis``).
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
                "%s timeout (attempt %s/%s, timeout_s=%s): %s",
                log_label,
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
                "%s HTTP %s (attempt %s/%s)%s: %s",
                log_label,
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
                "%s transport error (attempt %s/%s): %s",
                log_label,
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
            f"{log_label} request timed out after {cfg.retry_attempts} attempt(s) "
            f"(timeout_s={cfg.timeout_s})."
        ) from last_exc
    raise last_exc
