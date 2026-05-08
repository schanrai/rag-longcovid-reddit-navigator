"""HTTP routes for Phase 6 API."""
from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse

from src.api.dependencies import (
    RetrievalConfigDep,
    SettingsDep,
    SynthesisConfigDep,
    WeaviateClientDep,
)
from src.api.exceptions import ApiError
from src.api.models import (
    ClarificationResponse,
    HealthResponse,
    QueryRequest,
    QuerySuccessResponse,
)
from src.api.service import execute_query

log = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(settings: SettingsDep) -> HealthResponse:
    return HealthResponse(status="ok", service=settings.service_name)


@router.post("/query", response_model=None)
def post_query(
    request: Request,
    response: Response,
    body: QueryRequest,
    settings: SettingsDep,
    cfg: RetrievalConfigDep,
    synth_cfg: SynthesisConfigDep,
    weaviate_client: WeaviateClientDep,
) -> QuerySuccessResponse | ClarificationResponse | JSONResponse:
    request_id = str(uuid.uuid4())
    response.headers["X-Request-ID"] = request_id
    log.info(
        "query start request_id=%s (service=%s)",
        request_id,
        settings.service_name,
    )
    try:
        result = execute_query(
            body,
            cfg=cfg,
            synth_cfg=synth_cfg,
            weaviate_client=weaviate_client,
            request_id=request_id,
        )
        if isinstance(result, ClarificationResponse):
            log.info("query clarification request_id=%s", request_id)
        else:
            log.info(
                "query ok request_id=%s chunks=%d cited=%d",
                request_id,
                result.metadata.chunks_retrieved,
                result.metadata.chunks_cited,
            )
        return result
    except ApiError as exc:
        log.warning(
            "query error request_id=%s status=%s body=%s",
            request_id,
            exc.status_code,
            exc.body,
        )
        return JSONResponse(status_code=exc.status_code, content=exc.body)
