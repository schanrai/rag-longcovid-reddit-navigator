"""
FastAPI dependencies — config, Weaviate client from ``app.state``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated

import weaviate
from fastapi import Depends, Request

from src.retrieval.config import RetrievalConfig
from src.synthesis import SynthesisConfig


@dataclass(frozen=True, slots=True)
class APISettings:
    """Environment-driven API settings."""

    cors_origins: tuple[str, ...]
    service_name: str

    @classmethod
    def load(cls) -> APISettings:
        raw = os.environ.get("CORS_ORIGINS", "http://localhost:3000").strip()
        origins = tuple(x.strip() for x in raw.split(",") if x.strip())
        if not origins:
            origins = ("http://localhost:3000",)
        name = os.environ.get("API_SERVICE_NAME", "long-covid-compass-api").strip() or "long-covid-compass-api"
        return cls(cors_origins=origins, service_name=name)


@lru_cache
def get_api_settings() -> APISettings:
    return APISettings.load()


def get_weaviate_client(request: Request) -> weaviate.WeaviateClient:
    client = getattr(request.app.state, "weaviate_client", None)
    if client is None:
        raise RuntimeError("Weaviate client not initialized (lifespan)")
    return client


def get_retrieval_config() -> RetrievalConfig:
    cfg = RetrievalConfig()
    cfg.reranker.enabled = True
    return cfg


def get_synthesis_config() -> SynthesisConfig:
    return SynthesisConfig()


WeaviateClientDep = Annotated[weaviate.WeaviateClient, Depends(get_weaviate_client)]
RetrievalConfigDep = Annotated[RetrievalConfig, Depends(get_retrieval_config)]
SynthesisConfigDep = Annotated[SynthesisConfig, Depends(get_synthesis_config)]
SettingsDep = Annotated[APISettings, Depends(get_api_settings)]
