"""
Pydantic wire models for POST /query and GET /health.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from src.api.citations import CitedSource
from src.api.policy_block import PolicyBlockPayload

FailedStage = Literal["rewriting", "searching", "reading", "synthesizing"]


class QueryRequest(BaseModel):
    """
    POST /query body.

    Initial ask: ``{"query": "..."}`` only.

    Clarification resolution: ``original_query`` + ``selected_rewrite_index`` **or**
    ``edited_query`` (re-run rewriter on new text).
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=8000,
        description="User query (initial) or placeholder when resolving via edited_query only",
    )
    original_query: str | None = Field(
        default=None,
        max_length=8000,
        description="Original query from the clarification step; required with selected_rewrite_index",
    )
    selected_rewrite_index: int | None = Field(
        default=None,
        ge=0,
        description="0-based index into rewriter candidates for original_query",
    )
    edited_query: str | None = Field(
        default=None,
        max_length=8000,
        description="Free-text from 'Other'; triggers a fresh rewriter pass",
    )

    @model_validator(mode="after")
    def _resolution_fields(self) -> QueryRequest:
        has_idx = self.selected_rewrite_index is not None
        has_edit = self.edited_query is not None and self.edited_query.strip() != ""
        if has_idx and has_edit:
            raise ValueError("Use either selected_rewrite_index or edited_query, not both.")
        if has_idx and not (self.original_query and self.original_query.strip()):
            raise ValueError("original_query is required when selected_rewrite_index is set.")
        return self


class QueryMetadata(BaseModel):
    latency_ms: int = Field(..., ge=0)
    chunks_retrieved: int = Field(..., ge=0)
    chunks_cited: int = Field(..., ge=0)
    reranker_used: bool
    model: str
    rewrite_latency_ms: int | None = None
    retrieval_latency_ms: int | None = None
    synthesis_latency_ms: int | None = None


class QuerySuccessResponse(BaseModel):
    policy_block: PolicyBlockPayload
    answer_markdown: str
    sources: list[CitedSource]
    rewritten_query: str
    original_query: str
    metadata: QueryMetadata


class RewriteCandidateWire(BaseModel):
    query: str
    explanation: str
    confidence: float


class ClarificationResponse(BaseModel):
    """First-step response when rewriter returns clarification mode."""

    mode: Literal["clarification"] = "clarification"
    intent: str
    rewrites: list[RewriteCandidateWire]
    original_query: str


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str = "long-covid-compass-api"


class ErrorBody(BaseModel):
    code: str
    message: str


class QueryErrorResponse(BaseModel):
    error: ErrorBody
    failed_stage: FailedStage | None = None
