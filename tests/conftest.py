"""Shared pytest fixtures."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import get_weaviate_client
from src.api.main import create_app


@pytest.fixture
def client() -> TestClient:
    """
    FastAPI app without Weaviate connect or reranker warmup (fast).

    Injects a dummy Weaviate client so ``POST /query`` can run through validation
    and mocked pipeline paths without ``connect_weaviate=True``.
    """
    app = create_app(connect_weaviate=False, warmup_reranker=False)
    app.dependency_overrides[get_weaviate_client] = lambda: MagicMock()
    return TestClient(app)
