"""
FastAPI application — Long Covid Compass backend.

Run:
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.dependencies import APISettings
from src.api.routes import router
from src.retrieval.config import RetrievalConfig
from src.retrieval.hybrid_search import _build_weaviate_client
from src.retrieval.reranker import warmup_cross_encoder

log = logging.getLogger(__name__)


def _configure_logging() -> None:
    if not logging.getLogger().handlers:
        level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    _configure_logging()
    settings: APISettings = app.state.api_settings
    connect = getattr(app.state, "_connect_weaviate", True)
    do_warmup = getattr(app.state, "_warmup_reranker", True)
    if connect:
        log.info("Connecting Weaviate (service=%s)", settings.service_name)
        app.state.weaviate_client = _build_weaviate_client()
    else:
        log.warning("Skipping Weaviate connect (test / offline mode)")
        app.state.weaviate_client = None

    rcfg = RetrievalConfig()
    rcfg.reranker.enabled = True
    if do_warmup:
        warmup_cross_encoder(rcfg)

    yield

    client = getattr(app.state, "weaviate_client", None)
    if client is not None:
        try:
            client.close()
        except Exception as exc:
            log.warning("Weaviate close: %s", exc)


def create_app(*, connect_weaviate: bool = True, warmup_reranker: bool = True) -> FastAPI:
    """
    Application factory.

    Parameters
    ----------
    connect_weaviate:
        When False, lifespan does not open Weaviate (unit tests override client dep).
    warmup_reranker:
        When False, skip cross-encoder load in lifespan (faster tests).
    """
    settings = APISettings.load()
    app = FastAPI(
        title="Long Covid Compass API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.api_settings = settings
    app.state._connect_weaviate = connect_weaviate
    app.state._warmup_reranker = warmup_reranker

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app(connect_weaviate=True, warmup_reranker=True)
