"""Unit tests for rewriter generateContent args. No live model calls."""
from __future__ import annotations

from typing import Any

import pytest

from src.retrieval.config import RetrievalConfig
from src.retrieval.query_rewriter import rewrite


def test_rewrite_sends_model_and_minimal_thinking(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _fake_generate_text(**kwargs: Any) -> str:
        captured.update(kwargs)
        return (
            '{"mode":"confident","intent":"symptom",'
            '"rewrites":[{"query":"Long COVID fatigue","explanation":"x","confidence":0.95}]}'
        )

    monkeypatch.setattr("src.retrieval.query_rewriter.generate_text", _fake_generate_text)

    result = rewrite("tired", cfg=RetrievalConfig(), api_key="test-key")

    assert captured["model"] == "gemini-3-flash-preview"
    assert captured["thinking_level"] == "minimal"
    assert captured["max_output_tokens"] == 512
    assert result.rewrites[0].query == "Long COVID fatigue"
