"""Input gate (Layer 2) — no Weaviate required."""
from __future__ import annotations

from fastapi.testclient import TestClient


def test_query_rejects_nonsense(client: TestClient) -> None:
    r = client.post("/query", json={"query": ";;;"})
    assert r.status_code == 400
    body = r.json()
    assert body["error"]["code"] == "query_nonsense"
    assert body["failed_stage"] is None


def test_query_rejects_whitespace_only(client: TestClient) -> None:
    r = client.post("/query", json={"query": "   "})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "query_empty"
