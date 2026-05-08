"""
Phase 2 QA gate — clarification mode: first response shape, second-call resolution, 400s.

Uses patched ``rewrite`` / ``retrieve_from_rewrite`` / ``generate_synthesis`` so tests stay offline.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import get_weaviate_client
from src.api.main import create_app
from src.retrieval.models import (
    ChunkMetadata,
    IntentCategory,
    RankingPreset,
    RetrievalResult,
    RewriteCandidate,
    RewriteMode,
    RewriteResult,
    SearchResult,
)


def _client() -> TestClient:
    app = create_app(connect_weaviate=False, warmup_reranker=False)
    app.dependency_overrides[get_weaviate_client] = lambda: MagicMock()
    return TestClient(app)


def _clarification(oq: str) -> RewriteResult:
    return RewriteResult(
        mode=RewriteMode.CLARIFICATION,
        original_query=oq,
        rewrites=[
            RewriteCandidate(query="opt-a keywords", explanation="a", confidence=0.5),
            RewriteCandidate(query="opt-b keywords", explanation="b", confidence=0.48),
        ],
        intent=IntentCategory.SYMPTOM,
    )


def test_phase2_first_response_no_synthesis_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.api.service.rewrite",
        lambda q, **k: _clarification(q),
    )
    with _client() as client:
        r = client.post(
            "/query",
            json={"query": "LC awareness doctor family"},
        )
    assert r.status_code == 200
    data = r.json()
    assert data["mode"] == "clarification"
    assert set(data.keys()) == {"mode", "intent", "rewrites", "original_query"}
    for bad in ("policy_block", "answer_markdown", "sources", "metadata"):
        assert bad not in data
    assert len(data["rewrites"]) == 2
    assert data["rewrites"][0]["query"] == "opt-a keywords"


def test_phase2_resolve_with_selected_index_then_success_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def rw(q: str, **k: object) -> RewriteResult:
        return _clarification(q)

    monkeypatch.setattr("src.api.service.rewrite", rw)

    def one_hit(rr: RewriteResult, *a: object, **k: object) -> RetrievalResult:
        assert rr.best_rewrite.query == "opt-b keywords"
        sr = SearchResult(
            chunk_id="c1",
            text=("Discussion " * 8).strip(),
            hybrid_score=0.9,
            final_score=0.9,
            metadata=ChunkMetadata(
                chunk_id="c1",
                chunk_type="comment",
                permalink="/r/x/y",
                created_utc=1_700_000_000,
                comment_score=1,
                num_comments=2,
            ),
        )
        return RetrievalResult(
            query=rr,
            results=[sr],
            preset=RankingPreset.MOST_RELEVANT,
            reranker_enabled=True,
            elapsed_ms=1.0,
        )

    monkeypatch.setattr("src.api.service.retrieve_from_rewrite", one_hit)

    def synth_stub(retrieval: RetrievalResult, **_: object):
        return SimpleNamespace(
            answer="**T**\n\nHello [1].\n\nDisclaimer.",
            metadata=SimpleNamespace(model="google/gemini-3-flash-preview"),
        )

    monkeypatch.setattr("src.api.service.generate_synthesis", synth_stub)

    with _client() as client:
        r2 = client.post(
            "/query",
            json={
                "query": "placeholder",
                "original_query": "LC awareness doctor family",
                "selected_rewrite_index": 1,
            },
        )
    assert r2.status_code == 200
    body = r2.json()
    assert "policy_block" in body
    assert body["rewritten_query"] == "opt-b keywords"
    assert body["sources"][0]["n"] == 1


def test_phase2_resolve_with_edited_query(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def rw(q: str, **k: object) -> RewriteResult:
        seen.append(q)
        return RewriteResult(
            mode=RewriteMode.CONFIDENT,
            original_query=q,
            rewrites=[
                RewriteCandidate(
                    query=f"rw-{q[:20]}",
                    explanation="e",
                    confidence=0.95,
                ),
            ],
            intent=IntentCategory.TREATMENT,
        )

    monkeypatch.setattr("src.api.service.rewrite", rw)

    monkeypatch.setattr(
        "src.api.service.retrieve_from_rewrite",
        lambda rr, *a, **k: RetrievalResult(
            query=rr,
            results=[],
            preset=RankingPreset.MOST_RELEVANT,
            reranker_enabled=True,
            elapsed_ms=1.0,
        ),
    )

    with _client() as client:
        r = client.post(
            "/query",
            json={
                "query": "x",
                "edited_query": "  PEM fatigue Long COVID  ",
            },
        )
    assert r.status_code == 200
    assert seen == ["PEM fatigue Long COVID"]


def test_phase2_invalid_rewrite_index_400(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.api.service.rewrite",
        lambda q, **k: _clarification(q),
    )
    with _client() as client:
        r = client.post(
            "/query",
            json={
                "query": "x",
                "original_query": "LC awareness doctor family",
                "selected_rewrite_index": 5,
            },
        )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_rewrite_index"


def test_phase2_edited_query_whitespace_only_400() -> None:
    with _client() as client:
        r = client.post(
            "/query",
            json={
                "query": "What helps brain fog?",
                "edited_query": "   \t  ",
            },
        )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "edited_query_empty"


def test_phase2_edited_query_empty_string_400() -> None:
    with _client() as client:
        r = client.post(
            "/query",
            json={"query": "What helps brain fog?", "edited_query": ""},
        )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "edited_query_empty"
