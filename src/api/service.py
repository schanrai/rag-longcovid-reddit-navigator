"""
POST /query orchestration: rewrite → retrieve → policy strip → cited sources → response.
"""
from __future__ import annotations

import logging
import time
from typing import Union

import httpx
import weaviate

from src.api.citations import build_cited_sources
from src.api.exceptions import ApiError
from src.api.input_gate import validate_query_text
from src.api.models import (
    ClarificationResponse,
    QueryMetadata,
    QueryRequest,
    QuerySuccessResponse,
    RewriteCandidateWire,
)
from src.api.policy_block import PolicyBlockPayload, extract_policy_block
from src.retrieval.config import RetrievalConfig
from src.retrieval.models import RewriteCandidate, RewriteMode, RewriteResult
from src.retrieval.pipeline import retrieve_from_rewrite
from src.retrieval.query_rewriter import rewrite
from src.synthesis import SynthesisConfig, generate_synthesis

log = logging.getLogger(__name__)

ResponseUnion = Union[QuerySuccessResponse, ClarificationResponse]


def _text_for_input_gate(body: QueryRequest) -> str:
    if body.edited_query is not None and body.edited_query.strip():
        return body.edited_query.strip()
    if body.selected_rewrite_index is not None:
        return (body.original_query or "").strip()
    return body.query.strip()


def _rewrite_for_request(body: QueryRequest, *, cfg: RetrievalConfig) -> RewriteResult:
    """Apply clarification resolution rules, then call rewriter."""
    if body.edited_query is not None and body.edited_query.strip():
        return rewrite(body.edited_query.strip(), cfg=cfg)

    if body.selected_rewrite_index is not None:
        oq = (body.original_query or "").strip()
        rr = rewrite(oq, cfg=cfg)
        if rr.mode != RewriteMode.CLARIFICATION:
            return rr
        idx = body.selected_rewrite_index
        if idx < 0 or idx >= len(rr.rewrites):
            raise ApiError(
                400,
                code="invalid_rewrite_index",
                message="That option is no longer valid. Please start a new search.",
                failed_stage=None,
            )
        chosen = rr.rewrites[idx]
        return RewriteResult(
            mode=RewriteMode.CONFIDENT,
            original_query=oq,
            rewrites=[
                RewriteCandidate(
                    query=chosen.query,
                    explanation=chosen.explanation,
                    confidence=max(chosen.confidence, 0.99),
                )
            ],
            intent=rr.intent,
        )

    return rewrite(body.query.strip(), cfg=cfg)


def execute_query(
    body: QueryRequest,
    *,
    cfg: RetrievalConfig,
    synth_cfg: SynthesisConfig,
    weaviate_client: weaviate.WeaviateClient,
    request_id: str | None = None,
) -> ResponseUnion:
    """
    Full POST /query handling (sync). Raises ``ApiError`` for error envelopes.

    ``request_id`` is logged on structured ``query_stages`` lines for log aggregation.
    """
    rid = request_id or "-"
    if body.edited_query is not None and not body.edited_query.strip():
        raise ApiError(
            400,
            code="edited_query_empty",
            message="Please enter a rephrased question before searching.",
            failed_stage=None,
        )

    gate_text = _text_for_input_gate(body)
    gate = validate_query_text(gate_text)
    if not gate.ok:
        raise ApiError(
            400,
            code=gate.error_code,
            message=gate.message,
            failed_stage=None,
        )

    t_rewrite0 = time.perf_counter()
    try:
        rewrite_result = _rewrite_for_request(body, cfg=cfg)
    except ApiError:
        raise
    except ValueError as exc:
        log.warning("rewrite validation error: %s", exc)
        raise ApiError(
            400,
            code="rewrite_error",
            message="Please rephrase as a normal question about Long COVID.",
            failed_stage=None,
        ) from exc
    except httpx.HTTPError as exc:
        log.exception("rewrite HTTP error")
        raise ApiError(
            502,
            code="upstream_error",
            message="Something went wrong while putting your answer together. Let's try again.",
            failed_stage="rewriting",
        ) from exc
    except Exception as exc:
        log.exception("rewrite unexpected error")
        raise ApiError(
            502,
            code="pipeline_error",
            message="Something went wrong while putting your answer together. Let's try again.",
            failed_stage="rewriting",
        ) from exc
    rewrite_ms = int((time.perf_counter() - t_rewrite0) * 1000)

    if rewrite_result.mode == RewriteMode.CLARIFICATION:
        log.info(
            "query_stages request_id=%s rewrite_ms=%d retrieval_ms=- synthesis_ms=- "
            "total_ms=%d outcome=clarification",
            rid,
            rewrite_ms,
            rewrite_ms,
        )
        return ClarificationResponse(
            mode="clarification",
            intent=rewrite_result.intent.value,
            rewrites=[
                RewriteCandidateWire(
                    query=c.query,
                    explanation=c.explanation,
                    confidence=c.confidence,
                )
                for c in rewrite_result.rewrites
            ],
            original_query=rewrite_result.original_query,
        )

    search_query = rewrite_result.best_rewrite.query
    t_retrieve0 = time.perf_counter()
    try:
        retrieval = retrieve_from_rewrite(
            rewrite_result,
            search_query=search_query,
            cfg=cfg,
            weaviate_client=weaviate_client,
        )
    except httpx.HTTPError as exc:
        log.exception("retrieval HTTP error")
        raise ApiError(
            502,
            code="upstream_error",
            message="Something went wrong while putting your answer together. Let's try again.",
            failed_stage="searching",
        ) from exc
    except EnvironmentError as exc:
        log.exception("retrieval configuration error")
        raise ApiError(
            503,
            code="service_unavailable",
            message="Something went wrong while putting your answer together. Let's try again.",
            failed_stage="searching",
        ) from exc
    except Exception as exc:
        log.exception("retrieval unexpected error")
        # Distinguish rerank vs search heuristically via message is fragile; use "reading" for rank/rerank.
        stage = "searching"
        msg = str(exc).lower()
        if "rerank" in msg or "cross-encoder" in msg or "sentence_transformers" in msg:
            stage = "reading"
        raise ApiError(
            502,
            code="pipeline_error",
            message="Something went wrong while putting your answer together. Let's try again.",
            failed_stage=stage,
        ) from exc

    retrieval_ms = int((time.perf_counter() - t_retrieve0) * 1000)

    if not retrieval.results:
        total_empty = rewrite_ms + retrieval_ms
        log.info(
            "query_stages request_id=%s rewrite_ms=%d retrieval_ms=%d synthesis_ms=0 "
            "total_ms=%d chunks_retrieved=0 chunks_cited=0 outcome=empty_retrieval",
            rid,
            rewrite_ms,
            retrieval_ms,
            total_empty,
        )
        return QuerySuccessResponse(
            policy_block=PolicyBlockPayload(type=None, markdown=""),
            answer_markdown="",
            sources=[],
            rewritten_query=rewrite_result.best_rewrite.query,
            original_query=rewrite_result.original_query,
            metadata=QueryMetadata(
                latency_ms=total_empty,
                chunks_retrieved=0,
                chunks_cited=0,
                reranker_used=cfg.reranker.enabled,
                model=synth_cfg.model,
                rewrite_latency_ms=rewrite_ms,
                retrieval_latency_ms=retrieval_ms,
                synthesis_latency_ms=0,
            ),
        )

    t_synth0 = time.perf_counter()
    try:
        synth = generate_synthesis(retrieval, cfg=synth_cfg)
    except httpx.HTTPError as exc:
        log.exception("synthesis HTTP error")
        raise ApiError(
            502,
            code="upstream_error",
            message="Something went wrong while putting your answer together. Let's try again.",
            failed_stage="synthesizing",
        ) from exc
    except Exception as exc:
        log.exception("synthesis error")
        raise ApiError(
            502,
            code="pipeline_error",
            message="Something went wrong while putting your answer together. Let's try again.",
            failed_stage="synthesizing",
        ) from exc
    synth_ms = int((time.perf_counter() - t_synth0) * 1000)

    policy_block, cleaned_answer = extract_policy_block(synth.answer)
    max_ctx = synth_cfg.max_chunks_in_context
    context_chunks = retrieval.results[:max_ctx]
    cited_sources, _orphans = build_cited_sources(cleaned_answer, context_chunks)

    total_ms = rewrite_ms + retrieval_ms + synth_ms
    log.info(
        "query_stages request_id=%s rewrite_ms=%d retrieval_ms=%d synthesis_ms=%d "
        "total_ms=%d chunks_retrieved=%d chunks_cited=%d outcome=success",
        rid,
        rewrite_ms,
        retrieval_ms,
        synth_ms,
        total_ms,
        len(retrieval.results),
        len(cited_sources),
    )
    return QuerySuccessResponse(
        policy_block=policy_block,
        answer_markdown=cleaned_answer,
        sources=cited_sources,
        rewritten_query=rewrite_result.best_rewrite.query,
        original_query=rewrite_result.original_query,
        metadata=QueryMetadata(
            latency_ms=total_ms,
            chunks_retrieved=len(retrieval.results),
            chunks_cited=len(cited_sources),
            reranker_used=retrieval.reranker_enabled,
            model=synth.metadata.model,
            rewrite_latency_ms=rewrite_ms,
            retrieval_latency_ms=retrieval_ms,
            synthesis_latency_ms=synth_ms,
        ),
    )
