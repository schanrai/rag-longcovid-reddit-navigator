"""Unit tests for policy_block extraction (golden-style snippets)."""
from __future__ import annotations

from src.api.policy_block import extract_policy_block, split_policy_prefix


def test_no_heading_entire_body_is_synthesis() -> None:
    raw = "**Fatigue**\n\nSome text [1]."
    prefix, body = split_policy_prefix(raw)
    assert prefix == ""
    assert body == raw


def test_emergency_prefix_before_topic() -> None:
    raw = (
        "If you have crushing chest pain right now, seek emergency care immediately. "
        "This tool cannot determine whether symptoms are just Long COVID.\n\n"
        "**Community experiences**\n\nPeers report [1]."
    )
    policy, cleaned = extract_policy_block(raw)
    assert policy.markdown != ""
    assert policy.type == "emergency"
    assert cleaned.startswith("**Community experiences**")
    assert "crushing chest" not in cleaned


def test_prescriber_prefix_classification() -> None:
    raw = (
        "Stopping or changing prescription medication must be decided with your prescribing physician. "
        "The summary below is peer-reported context only.\n\n"
        "**Treatments**\n\nSome discussion [1]."
    )
    policy, cleaned = extract_policy_block(raw)
    assert policy.type == "prescriber"
    assert "**Treatments**" in cleaned


def test_q23_style_beta_blocker_snippet() -> None:
    raw = (
        "Stopping beta blockers without medical supervision can be dangerous. "
        "Any medication change should be discussed with your prescribing clinician.\n\n"
        "**Beta blockers and Long COVID**\n\nCommunity posts mention [1]."
    )
    policy, cleaned = extract_policy_block(raw)
    assert policy.markdown
    assert cleaned.startswith("**Beta blockers")
