"""Clarification resolution: edited_query and selected_rewrite_index."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.api.models import QueryRequest
from src.api.service import execute_query
from src.synthesis import SynthesisConfig
from src.retrieval.config import RetrievalConfig
from src.retrieval.models import (
    IntentCategory,
    RankingPreset,
    RetrievalResult,
    RewriteCandidate,
    RewriteMode,
    RewriteResult,
)


def _confident_result(q: str) -> RewriteResult:
    return RewriteResult(
        mode=RewriteMode.CONFIDENT,
        original_query=q,
        rewrites=[
            RewriteCandidate(query=f"rewritten-{q}", explanation="e", confidence=0.95),
        ],
        intent=IntentCategory.TREATMENT,
    )


def _clarification_result(oq: str) -> RewriteResult:
    return RewriteResult(
        mode=RewriteMode.CLARIFICATION,
        original_query=oq,
        rewrites=[
            RewriteCandidate(query="opt0", explanation="a", confidence=0.5),
            RewriteCandidate(query="opt1", explanation="b", confidence=0.45),
        ],
        intent=IntentCategory.SYMPTOM,
    )


def test_edited_query_uses_fresh_rewrite(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def rw(q: str, **kwargs: object) -> RewriteResult:
        seen.append(q)
        return _confident_result(q)

    monkeypatch.setattr("src.api.service.rewrite", rw)

    def empty_retrieval(rr: RewriteResult, *a: object, **k: object) -> RetrievalResult:
        return RetrievalResult(
            query=rr,
            results=[],
            preset=RankingPreset.MOST_RELEVANT,
            reranker_enabled=True,
            elapsed_ms=1.0,
        )

    monkeypatch.setattr("src.api.service.retrieve_from_rewrite", empty_retrieval)

    wv = MagicMock()
    body = QueryRequest(query="ignored", edited_query="  PEM and fatigue  ")
    execute_query(body, cfg=RetrievalConfig(), synth_cfg=SynthesisConfig(), weaviate_client=wv)
    assert seen == ["PEM and fatigue"]


def test_selected_rewrite_index_picks_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    def rw(q: str, **kwargs: object) -> RewriteResult:
        if q == "ambiguous thing":
            return _clarification_result("ambiguous thing")
        return _confident_result(q)

    monkeypatch.setattr("src.api.service.rewrite", rw)

    def empty_retrieval(rr: RewriteResult, *a: object, **k: object) -> RetrievalResult:
        assert rr.best_rewrite.query == "opt1"
        return RetrievalResult(
            query=rr,
            results=[],
            preset=RankingPreset.MOST_RELEVANT,
            reranker_enabled=True,
            elapsed_ms=1.0,
        )

    monkeypatch.setattr("src.api.service.retrieve_from_rewrite", empty_retrieval)

    body = QueryRequest(
        query="x",
        original_query="ambiguous thing",
        selected_rewrite_index=1,
    )
    execute_query(body, cfg=RetrievalConfig(), synth_cfg=SynthesisConfig(), weaviate_client=MagicMock())


def test_invalid_rewrite_index_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.api.service.rewrite", lambda q, **k: _clarification_result(q))

    from src.api.exceptions import ApiError

    body = QueryRequest(
        query="x",
        original_query="ambiguous thing",
        selected_rewrite_index=99,
    )
    with pytest.raises(ApiError) as excinfo:
        execute_query(body, cfg=RetrievalConfig(), synth_cfg=SynthesisConfig(), weaviate_client=MagicMock())
    assert excinfo.value.status_code == 400
    assert excinfo.value.body["error"]["code"] == "invalid_rewrite_index"
