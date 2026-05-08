"""
Extract synthesis safety / prescriber prefix blocks into API ``policy_block`` (Phase 6 Q4).

Order of operations: run on raw ``answer_markdown`` from synthesis, **before** ``[n]`` parsing.
"""
from __future__ import annotations

import logging
import re
from typing import Final, Literal

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

# First **Topic Heading** line — same structural signal as synthesis_system_prompt.txt
_TOPIC_HEADING_LINE: Final[re.Pattern[str]] = re.compile(r"^\*\*.+\*\*\s*$")
# Policy/safety blocks must not use [n] anchors; prefix text with anchors is synthesis intro.
_PREFIX_HAS_CITATION: Final[re.Pattern[str]] = re.compile(r"\[\d+\]")

PolicyBlockType = Literal["emergency", "prescriber"] | None


class PolicyBlockPayload(BaseModel):
    """Wire payload for ``policy_block`` on POST /query success."""

    type: PolicyBlockType = Field(default=None, description="emergency | prescriber | null")
    markdown: str = Field(default="", description="Policy prose only; empty when absent")


_EMERGENCY_HINTS: Final[tuple[str, ...]] = (
    "emergency",
    "911",
    "emergency room",
    "urgent",
    "crushing chest",
    "chest pain",
    "immediate medical",
    "in-person",
    "cannot determine",
    "just long covid",
)

_PRESCRIBER_HINTS: Final[tuple[str, ...]] = (
    "prescribing physician",
    "prescribing clinician",
    "qualified clinician",
    "prescription medication",
    "taper",
    "stopping, starting, or changing",
    "peer-reported context",
    "not instructions",
)


def split_policy_prefix(raw_answer_markdown: str) -> tuple[str, str]:
    """
    Split raw synthesis markdown into (policy_prefix, synthesis_body).

    If no topic heading line is found, returns ("", raw trimmed) — entire string is body.
    """
    text = (raw_answer_markdown or "").strip()
    if not text:
        return "", ""

    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if _TOPIC_HEADING_LINE.match(stripped):
            prefix = "\n".join(lines[:i]).strip()
            body = "\n".join(lines[i:]).strip()
            return prefix, body

    return "", text


def classify_policy_prefix(prefix: str) -> PolicyBlockType:
    """Best-effort classification from prefix text (lowercased substring match)."""
    if not prefix.strip():
        return None
    low = prefix.lower()
    for hint in _EMERGENCY_HINTS:
        if hint in low:
            return "emergency"
    for hint in _PRESCRIBER_HINTS:
        if hint in low:
            return "prescriber"
    return None


def extract_policy_block(raw_answer_markdown: str) -> tuple[PolicyBlockPayload, str]:
    """
    Returns ``(policy_block, cleaned_answer_markdown)`` for API assembly.

    When a non-empty prefix exists before the first topic heading, it becomes
    ``policy_block.markdown`` and is stripped from the answer tab markdown.

    If that prefix contains ``[n]`` citation anchors, treat it as synthesis (model
    wrote cited intro before the first topic heading); return the full raw string
    as the answer body and an empty policy block so citation parsing stays correct.
    """
    text = (raw_answer_markdown or "").strip()
    prefix, body = split_policy_prefix(raw_answer_markdown)
    if not prefix:
        return PolicyBlockPayload(type=None, markdown=""), body

    if _PREFIX_HAS_CITATION.search(prefix):
        log.debug(
            "policy_block: prefix before first **Topic** contains [n]; "
            "skipping policy strip (prefix_len=%d)",
            len(prefix),
        )
        return PolicyBlockPayload(type=None, markdown=""), text

    ptype = classify_policy_prefix(prefix)
    if ptype is None:
        log.warning(
            "policy_block prefix present but type unclassified (length=%d chars)",
            len(prefix),
        )

    return PolicyBlockPayload(type=ptype, markdown=prefix), body
