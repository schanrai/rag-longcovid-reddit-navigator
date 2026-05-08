"""
Phase 3 QA gate (fastapi_phase_gates plan) — automated slice.

Manual checklist: docs/deploy-smoke-phase3.md

Run:
    pytest tests/test_phase3_production_readiness.py -v
"""
from __future__ import annotations

import logging
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


def _offline_client() -> TestClient:
    app = create_app(connect_weaviate=False, warmup_reranker=False)
    app.dependency_overrides[get_weaviate_client] = lambda: MagicMock()
    return TestClient(app)


def test_phase3_offline_app_starts_without_pipeline_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    QA: "Starts clean" — no pipeline env required when Weaviate is not connected
    (``create_app(connect_weaviate=False)`` + ``get_weaviate_client`` override).
    Lifespan must complete without missing-env crash.
    """
    for key in ("WEAVIATE_URL", "WEAVIATE_API_KEY", "OPENROUTER_API_KEY", "VOYAGE_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    with _offline_client() as client:
        r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_phase3_live_startup_requires_pipeline_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Live mode fails fast at startup if required keys are missing."""
    for key in ("WEAVIATE_URL", "WEAVIATE_API_KEY", "OPENROUTER_API_KEY", "VOYAGE_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(RuntimeError, match="Missing required environment variables"):
        with TestClient(create_app(connect_weaviate=True, warmup_reranker=False)):
            pass


def test_phase3_query_stages_log_line(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful POST /query emits a parseable query_stages line with per-stage ms."""
    caplog.set_level(logging.INFO)

    rr = RewriteResult(
        mode=RewriteMode.CONFIDENT,
        original_query="What helps brain fog?",
        rewrites=[
            RewriteCandidate(
                query="brain fog Long COVID",
                explanation="e",
                confidence=0.9,
            )
        ],
        intent=IntentCategory.TREATMENT,
    )
    sr = SearchResult(
        chunk_id="c1",
        text=("Discussion text " * 8).strip(),
        hybrid_score=0.9,
        final_score=0.9,
        metadata=ChunkMetadata(
            chunk_id="c1",
            chunk_type="comment",
            permalink="/r/LongCovid/x",
            created_utc=1_700_000_000,
            comment_score=1,
            num_comments=2,
        ),
    )
    retrieval = RetrievalResult(
        query=rr,
        results=[sr],
        preset=RankingPreset.MOST_RELEVANT,
        reranker_enabled=True,
        elapsed_ms=1.0,
    )

    monkeypatch.setattr("src.api.service.rewrite", lambda *a, **k: rr)
    monkeypatch.setattr("src.api.service.retrieve_from_rewrite", lambda *a, **k: retrieval)

    def synth_stub(*_a, **_k):
        return SimpleNamespace(
            answer="**Topic**\n\nPoint [1].\n\nDisclaimer.",
            metadata=SimpleNamespace(model="google/gemini-3-flash-preview"),
        )

    monkeypatch.setattr("src.api.service.generate_synthesis", synth_stub)

    with _offline_client() as client:
        r = client.post("/query", json={"query": "What helps brain fog?"})
    assert r.status_code == 200
    data = r.json()
    assert r.headers.get("X-Request-ID")
    md = data["metadata"]
    # QA: response carries structured latency fields (not only logs).
    for key in (
        "latency_ms",
        "rewrite_latency_ms",
        "retrieval_latency_ms",
        "synthesis_latency_ms",
        "chunks_retrieved",
        "chunks_cited",
        "reranker_used",
        "model",
    ):
        assert key in md
    assert md["rewrite_latency_ms"] is not None and md["rewrite_latency_ms"] >= 0
    assert md["retrieval_latency_ms"] is not None and md["retrieval_latency_ms"] >= 0
    assert md["synthesis_latency_ms"] is not None and md["synthesis_latency_ms"] >= 0
    assert md["latency_ms"] == (
        md["rewrite_latency_ms"] + md["retrieval_latency_ms"] + md["synthesis_latency_ms"]
    )

    joined = caplog.text
    assert "query_stages" in joined
    assert "rewrite_ms=" in joined
    assert "retrieval_ms=" in joined
    assert "synthesis_ms=" in joined
    assert "total_ms=" in joined
    assert "outcome=success" in joined
    assert "request_id=" in joined
