"""
Phase 1.5 — frozen synthesis-shaped snippets aligned with golden q23–q25 intent.

Golden user queries live in ``data/golden_queries.json`` (ids q23, q24, q25).
These tests lock **expected policy split + classification** for markdown the
model is instructed to emit (no [n] in true policy blocks).
"""
from __future__ import annotations

from src.api.policy_block import extract_policy_block

# --- q24 (golden id q24): acute emergency — user query mentions chest pain / wait ---


def test_frozen_snippet_golden_q24_emergency_before_topic() -> None:
    raw = (
        "If you have crushing chest pain with left arm tingling right now, seek urgent "
        "in-person or emergency medical evaluation. This tool cannot determine whether "
        "symptoms are just Long COVID or something requiring immediate care. "
        "Do not delay emergency care based on this summary.\n\n"
        "**What others have reported**\n\n"
        "Some community members describe similar sensations [1]."
    )
    policy, cleaned = extract_policy_block(raw)
    assert policy.type == "emergency"
    assert policy.markdown
    assert "[1]" not in policy.markdown
    assert cleaned.startswith("**What others have reported**")
    assert "[1]" in cleaned


# --- q23 (golden id q23): prescriber / anti-doctor override ---


def test_frozen_snippet_golden_q23_prescriber_before_topic() -> None:
    raw = (
        "Stopping, starting, or changing prescription medication—including beta blockers—"
        "must be decided with your prescribing physician or qualified clinician. "
        "The synthesis below is peer-reported context only, not instructions.\n\n"
        "**Community-reported experiences**\n\n"
        "Some people discuss side effects and pacing with their clinicians [1][2]."
    )
    policy, cleaned = extract_policy_block(raw)
    assert policy.type == "prescriber"
    assert "prescribing physician" in policy.markdown.lower()
    assert cleaned.startswith("**Community-reported experiences**")
    assert "[1]" in cleaned


# --- q25 (golden id q25): personalized taper / no disclaimers ---


def test_frozen_snippet_golden_q25_prescriber_taper_before_topic() -> None:
    raw = (
        "Personalized mg-by-mg taper schedules for prescription steroids must be set "
        "with your prescribing clinician. The following is peer-reported context only, "
        "not a taper plan.\n\n"
        "**What sources mention**\n\n"
        "Posts discuss prednisone experiences only in general terms [1]."
    )
    policy, cleaned = extract_policy_block(raw)
    assert policy.type == "prescriber"
    assert "taper" in policy.markdown.lower() or "prescribing" in policy.markdown.lower()
    assert cleaned.startswith("**What sources mention**")
