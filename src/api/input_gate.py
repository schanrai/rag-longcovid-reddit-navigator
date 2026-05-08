"""
Server-side input validation for POST /query (Layer 2 — authoritative).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

# Minimum alphabetic run length (tunable; validate against popular queries before locking).
MIN_ALPHABETIC_RUN: Final[int] = 3

_ALPHABETIC_RUN: Final[re.Pattern[str]] = re.compile(
    rf"[A-Za-z]{{{MIN_ALPHABETIC_RUN},}}"
)

# High-signal jailbreak / delimiter patterns (case-insensitive). Prefer structure over brittle bans.
_INJECTION_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"ignore\s+(previous|all|the)\s+instructions", re.I),
    re.compile(r"disregard\s+(the\s+)?(above|instructions)", re.I),
    re.compile(r"\bsystem\s*:", re.I),
    re.compile(r"you\s+are\s+now\s+", re.I),
    re.compile(r"\bDAN\b", re.I),
    re.compile(r"sudo\s+mode", re.I),
    re.compile(r"```(?:json|markdown|text)?", re.I),
)


@dataclass(frozen=True, slots=True)
class InputGateResult:
    ok: bool
    error_code: str
    message: str


def validate_query_text(query: str) -> InputGateResult:
    """
    Validate trimmed user query. Returns ok=False with stable ``error_code`` for 400 responses.
    """
    text = query.strip()
    if not text:
        return InputGateResult(False, "query_empty", "Please enter a question about Long COVID.")

    if len(text) > 8000:
        return InputGateResult(
            False,
            "query_too_long",
            "That question is too long. Please shorten it and try again.",
        )

    if not _ALPHABETIC_RUN.search(text):
        return InputGateResult(
            False,
            "query_nonsense",
            "Please rephrase as a normal question about Long COVID.",
        )

    for pat in _INJECTION_PATTERNS:
        if pat.search(text):
            return InputGateResult(
                False,
                "query_rejected_pattern",
                "Please rephrase as a normal question about Long COVID.",
            )

    # Unusual control characters (excluding common whitespace)
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", text):
        return InputGateResult(
            False,
            "query_control_chars",
            "Please rephrase as a normal question about Long COVID.",
        )

    return InputGateResult(True, "", "")
