"""Unit tests for Gemini response parsing and request shape. No live calls."""
from __future__ import annotations

from typing import Any

import pytest

from src.gemini_client import GeminiResponseError, extract_response_text, generate_text


def test_extract_skips_thought_part() -> None:
    body = {
        "candidates": [
            {
                "finishReason": "STOP",
                "content": {
                    "parts": [
                        {"text": "reasoning", "thought": True},
                        {"text": '{"answer":"ok"}'},
                    ]
                },
            }
        ]
    }
    assert extract_response_text(body) == '{"answer":"ok"}'


def test_extract_raises_when_candidates_missing() -> None:
    with pytest.raises(GeminiResponseError, match="candidates missing"):
        extract_response_text({})


def test_extract_raises_on_block_reason() -> None:
    with pytest.raises(GeminiResponseError, match="blockReason=SAFETY"):
        extract_response_text({"promptFeedback": {"blockReason": "SAFETY"}, "candidates": []})


def test_extract_raises_on_max_tokens_even_with_partial_text() -> None:
    body = {
        "candidates": [
            {
                "finishReason": "MAX_TOKENS",
                "content": {"parts": [{"text": '{"answer":'}]},
            }
        ]
    }
    with pytest.raises(GeminiResponseError, match="finishReason=MAX_TOKENS"):
        extract_response_text(body)


def test_extract_raises_on_safety_finish() -> None:
    body = {
        "candidates": [
            {
                "finishReason": "SAFETY",
                "content": {"parts": [{"text": "partial"}]},
            }
        ]
    }
    with pytest.raises(GeminiResponseError, match="finishReason=SAFETY"):
        extract_response_text(body)


def test_generate_text_posts_generate_content_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _Response:
        def json(self) -> dict[str, Any]:
            return {
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {"parts": [{"text": '{"answer":"ok"}'}]},
                    }
                ]
            }

    def _fake_post(**kwargs: Any) -> _Response:
        captured.update(kwargs)
        return _Response()

    monkeypatch.setattr("src.gemini_client.post_openrouter_chat_completions", _fake_post)

    text = generate_text(
        model="gemini-3-flash-preview",
        system_prompt="Reply with JSON only",
        user_prompt='{"answer":"ok"}',
        api_key="test-key",
        max_output_tokens=1800,
        temperature=0.0,
    )

    assert text == '{"answer":"ok"}'
    assert captured["url"].endswith("/models/gemini-3-flash-preview:generateContent")
    assert captured["headers"]["x-goog-api-key"] == "test-key"
    assert "Authorization" not in captured["headers"]
    assert captured["payload"]["systemInstruction"] == {"parts": [{"text": "Reply with JSON only"}]}
    assert captured["payload"]["contents"] == [
        {"role": "user", "parts": [{"text": '{"answer":"ok"}'}]}
    ]
    assert captured["payload"]["generationConfig"] == {"temperature": 0.0, "maxOutputTokens": 1800}


def test_generate_text_sends_thinking_level_only_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    payloads: list[dict[str, Any]] = []

    class _Response:
        def json(self) -> dict[str, Any]:
            return {
                "candidates": [
                    {"finishReason": "STOP", "content": {"parts": [{"text": "ok"}]}},
                ]
            }

    def _fake_post(**kwargs: Any) -> _Response:
        payloads.append(kwargs["payload"])
        return _Response()

    monkeypatch.setattr("src.gemini_client.post_openrouter_chat_completions", _fake_post)

    generate_text(
        model="gemini-3-flash-preview",
        user_prompt="hello",
        api_key="test-key",
        max_output_tokens=1800,
        thinking_level="low",
    )
    generate_text(
        model="gemini-3-flash-preview",
        user_prompt="hello",
        api_key="test-key",
        max_output_tokens=1800,
    )

    assert payloads[0]["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "low"}
    assert "thinkingConfig" not in payloads[1]["generationConfig"]


def test_generate_text_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY is not set"):
        generate_text(
            model="gemini-3-flash-preview",
            user_prompt="hello",
            max_output_tokens=1800,
        )
