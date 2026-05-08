"""
Phase 1 QA gate — baseline POST /query (confident path): envelopes, input gate, failed_stage.

Uses patched pipeline deps so tests stay offline and deterministic.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import httpx
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


def _client_app() -> TestClient:
    app = create_app(connect_weaviate=False, warmup_reranker=False)
    app.dependency_overrides[get_weaviate_client] = lambda: MagicMock()
    return TestClient(app)


def _confident_rewrite(q: str = "What helps brain fog?") -> RewriteResult:
    return RewriteResult(
        mode=RewriteMode.CONFIDENT,
        original_query=q,
        rewrites=[
            RewriteCandidate(query="brain fog Long COVID treatments community", explanation="e", confidence=0.92),
        ],
        intent=IntentCategory.TREATMENT,
    )


def _single_hit(rr: RewriteResult) -> RetrievalResult:
    sr = SearchResult(
        chunk_id="chunk_phase1_test",
        text=("Patient discussion text with enough substance for tests. " * 5).strip(),
        hybrid_score=0.9,
        final_score=0.9,
        metadata=ChunkMetadata(
            chunk_id="chunk_phase1_test",
            chunk_type="comment",
            permalink="/r/LongCovid/test",
            created_utc=1_700_000_000,
            comment_score=3,
            num_comments=42,
        ),
    )
    return RetrievalResult(
        query=rr,
        results=[sr],
        preset=RankingPreset.MOST_RELEVANT,
        reranker_enabled=True,
        elapsed_ms=50.0,
    )


def test_phase1_success_envelope_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    """200 confident path: required keys, nested shapes, metadata latency fields."""

    def synth_stub(retrieval: RetrievalResult, **_: object):
        from types import SimpleNamespace

        return SimpleNamespace(
            answer="**Topic heading**\n\nCited point [1].\n\nDisclaimer.",
            metadata=SimpleNamespace(model="google/gemini-3-flash-preview"),
        )

    monkeypatch.setattr("src.api.service.rewrite", lambda *a, **k: _confident_rewrite())
    monkeypatch.setattr(
        "src.api.service.retrieve_from_rewrite",
        lambda rr, **k: _single_hit(rr),
    )
    monkeypatch.setattr("src.api.service.generate_synthesis", synth_stub)

    client = _client_app()
    r = client.post("/query", json={"query": "What helps brain fog?"})
    assert r.status_code == 200
    data = r.json()

    assert data.keys() >= {
        "policy_block",
        "answer_markdown",
        "sources",
        "rewritten_query",
        "original_query",
        "metadata",
    }
    pb = data["policy_block"]
    assert set(pb.keys()) >= {"type", "markdown"}
    assert pb["type"] is None or pb["type"] in ("emergency", "prescriber")

    md = data["metadata"]
    for key in (
        "latency_ms",
        "chunks_retrieved",
        "chunks_cited",
        "reranker_used",
        "model",
        "rewrite_latency_ms",
        "retrieval_latency_ms",
        "synthesis_latency_ms",
    ):
        assert key in md
    assert md["chunks_retrieved"] >= 1
    assert md["rewrite_latency_ms"] is not None

    assert isinstance(data["sources"], list)
    if data["sources"]:
        src0 = data["sources"][0]
        for field in ("n", "text", "chunk_type", "permalink"):
            assert field in src0
        ns = [s["n"] for s in data["sources"]]
        assert ns == sorted(ns), "sources must be ordered by citation n ascending"


def test_phase1_empty_retrieval_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.api.service.rewrite", lambda *a, **k: _confident_rewrite())
    monkeypatch.setattr(
        "src.api.service.retrieve_from_rewrite",
        lambda rr, **k: RetrievalResult(
            query=rr,
            results=[],
            preset=RankingPreset.MOST_RELEVANT,
            reranker_enabled=True,
            elapsed_ms=10.0,
        ),
    )

    client = _client_app()
    r = client.post("/query", json={"query": "What helps brain fog?"})
    assert r.status_code == 200
    data = r.json()
    assert data["sources"] == []
    assert data["answer_markdown"] == ""
    assert data["metadata"]["chunks_retrieved"] == 0
    assert data["metadata"]["synthesis_latency_ms"] == 0


@pytest.mark.parametrize(
    "rewrite_exc",
    [
        pytest.param(httpx.ConnectError("rewrite transport"), id="connect_error"),
        pytest.param(httpx.TimeoutException("rewrite timeout"), id="timeout"),
    ],
)
def test_phase1_failed_stage_rewriting(
    monkeypatch: pytest.MonkeyPatch,
    rewrite_exc: Exception,
) -> None:
    def rewrite_raises(*a: object, **k: object) -> RewriteResult:
        raise rewrite_exc

    monkeypatch.setattr("src.api.service.rewrite", rewrite_raises)

    client = _client_app()
    r = client.post("/query", json={"query": "What helps brain fog?"})
    assert r.status_code == 502
    body = r.json()
    assert body["failed_stage"] == "rewriting"
    assert body["error"]["code"] in {"upstream_error", "pipeline_error"}


def test_phase1_failed_stage_searching(monkeypatch: pytest.MonkeyPatch) -> None:
    def retrieve_raises(_rr: RewriteResult, **k: object) -> RetrievalResult:
        raise httpx.ConnectError("weaviate")

    monkeypatch.setattr("src.api.service.rewrite", lambda *a, **k: _confident_rewrite())
    monkeypatch.setattr("src.api.service.retrieve_from_rewrite", retrieve_raises)

    client = _client_app()
    r = client.post("/query", json={"query": "What helps brain fog?"})
    assert r.status_code == 502
    assert r.json()["failed_stage"] == "searching"


def test_phase1_failed_stage_reading(monkeypatch: pytest.MonkeyPatch) -> None:
    def retrieve_raises(_rr: RewriteResult, **k: object) -> RetrievalResult:
        raise RuntimeError("cross-encoder rerank failed")

    monkeypatch.setattr("src.api.service.rewrite", lambda *a, **k: _confident_rewrite())
    monkeypatch.setattr("src.api.service.retrieve_from_rewrite", retrieve_raises)

    client = _client_app()
    r = client.post("/query", json={"query": "What helps brain fog?"})
    assert r.status_code == 502
    assert r.json()["failed_stage"] == "reading"


def test_phase1_failed_stage_synthesizing(monkeypatch: pytest.MonkeyPatch) -> None:
    def synth_raises(*a: object, **k: object) -> None:
        raise httpx.HTTPError("openrouter")

    monkeypatch.setattr("src.api.service.rewrite", lambda *a, **k: _confident_rewrite())
    monkeypatch.setattr(
        "src.api.service.retrieve_from_rewrite",
        lambda rr, **k: _single_hit(rr),
    )
    monkeypatch.setattr("src.api.service.generate_synthesis", synth_raises)

    client = _client_app()
    r = client.post("/query", json={"query": "What helps brain fog?"})
    assert r.status_code == 502
    assert r.json()["failed_stage"] == "synthesizing"
