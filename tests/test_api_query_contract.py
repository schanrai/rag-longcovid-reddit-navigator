"""Contract tests with mocked pipeline."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api.citations import CitedSource
from src.api.dependencies import get_weaviate_client
from src.api.main import create_app
from src.api.models import (
    ClarificationResponse,
    QueryMetadata,
    QuerySuccessResponse,
    RewriteCandidateWire,
)
from src.api.policy_block import PolicyBlockPayload


def _success_payload() -> QuerySuccessResponse:
    return QuerySuccessResponse(
        policy_block=PolicyBlockPayload(type=None, markdown=""),
        answer_markdown="**Topic**\n\nHello [1].\n\nDisclaimer here.",
        sources=[
            CitedSource(
                n=1,
                text="chunk",
                chunk_type="comment",
                comment_score=2,
                post_score=None,
                num_comments=10,
                post_summary=None,
                created_utc="2024-01-01T00:00:00+00:00",
                permalink="/r/x",
            )
        ],
        rewritten_query="rewritten",
        original_query="orig",
        metadata=QueryMetadata(
            latency_ms=100,
            chunks_retrieved=5,
            chunks_cited=1,
            reranker_used=True,
            model="google/gemini-3-flash-preview",
            rewrite_latency_ms=10,
            retrieval_latency_ms=40,
            synthesis_latency_ms=50,
        ),
    )


def test_query_success_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.api.routes.execute_query",
        lambda *a, **k: _success_payload(),
    )
    app = create_app(connect_weaviate=False, warmup_reranker=False)
    app.dependency_overrides[get_weaviate_client] = lambda: MagicMock()
    with TestClient(app) as client:
        r = client.post("/query", json={"query": "What helps brain fog?"})
    assert r.status_code == 200
    data = r.json()
    assert set(data.keys()) >= {
        "policy_block",
        "answer_markdown",
        "sources",
        "rewritten_query",
        "original_query",
        "metadata",
    }
    assert data["policy_block"]["type"] is None
    assert data["policy_block"]["markdown"] == ""
    assert data["sources"][0]["n"] == 1
    assert data["metadata"]["chunks_cited"] == 1


def test_query_error_failed_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.api.exceptions import ApiError

    def boom(*a, **k):
        raise ApiError(
            502,
            code="pipeline_error",
            message="Something went wrong while putting your answer together. Let's try again.",
            failed_stage="synthesizing",
        )

    monkeypatch.setattr("src.api.routes.execute_query", boom)
    app = create_app(connect_weaviate=False, warmup_reranker=False)
    app.dependency_overrides[get_weaviate_client] = lambda: MagicMock()
    with TestClient(app) as client:
        r = client.post("/query", json={"query": "What helps brain fog?"})
    assert r.status_code == 502
    body = r.json()
    assert body["failed_stage"] == "synthesizing"
    assert body["error"]["code"] == "pipeline_error"


def test_query_clarification_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    clar = ClarificationResponse(
        intent="symptom",
        rewrites=[
            RewriteCandidateWire(query="opt a", explanation="e", confidence=0.5),
            RewriteCandidateWire(query="opt b", explanation="e2", confidence=0.4),
        ],
        original_query="vague question",
    )

    monkeypatch.setattr("src.api.routes.execute_query", lambda *a, **k: clar)
    app = create_app(connect_weaviate=False, warmup_reranker=False)
    app.dependency_overrides[get_weaviate_client] = lambda: MagicMock()
    with TestClient(app) as client:
        r = client.post("/query", json={"query": "vague question"})
    assert r.status_code == 200
    data = r.json()
    assert data["mode"] == "clarification"
    assert "rewrites" in data
    assert len(data["rewrites"]) == 2
    assert "answer_markdown" not in data


def test_query_resolution_conflict_400(client: TestClient) -> None:
    r = client.post(
        "/query",
        json={
            "query": "x",
            "original_query": "a",
            "selected_rewrite_index": 0,
            "edited_query": "b",
        },
    )
    assert r.status_code == 422
