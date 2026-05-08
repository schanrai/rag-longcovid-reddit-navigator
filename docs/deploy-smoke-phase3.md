# Phase 3 QA gate — production readiness

Maps to **`fastapi_phase_gates`** plan → Phase 3 (warmup, env wiring, deploy, structured latency/logs).

---

## Automated

Run from repo root:

```bash
.venv/bin/python -m pytest tests/test_phase3_production_readiness.py -v
```

| Plan requirement | Test / artifact |
| --- | --- |
| **Starts clean** — no missing-env crash when using offline app + dependency overrides (Weaviate mocked) | `test_phase3_offline_app_starts_without_pipeline_env` — clears `WEAVIATE_*`, `OPENROUTER_API_KEY`, `VOYAGE_API_KEY`, boots `create_app(connect_weaviate=False, warmup_reranker=False)` with `get_weaviate_client` override, `GET /health` → 200 |
| **Live mode fails fast** if required env is missing (Railway / prod) | `test_phase3_live_startup_requires_pipeline_env` — `TestClient(create_app(connect_weaviate=True, …))` raises `RuntimeError` when keys are absent |
| **Log-shape sanity** — structured pipeline line | `test_phase3_query_stages_log_line` — log line contains `query_stages`, `rewrite_ms=`, `retrieval_ms=`, `synthesis_ms=`, `total_ms=`, `outcome=success`, `request_id=` |
| **Response metadata** — same latencies exposed to clients | Same test asserts JSON `metadata` includes `latency_ms`, per-stage ms, `chunks_*`, `model`, and `latency_ms` equals sum of stage ms; header `X-Request-ID` present |

---

## Manual (Railway + Weaviate Cloud)

Frontend (Vercel) is out of scope here.

### Preconditions

- Service variables set: `WEAVIATE_URL`, `WEAVIATE_API_KEY`, `OPENROUTER_API_KEY`, `VOYAGE_API_KEY`, `CORS_ORIGINS` (include your Vercel origin when testing from the browser).
- Start command from **`railway.toml`** / **`Procfile`**: `python -m uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`.

### Checklist

1. **Deploy / startup** — Logs show `lifespan weaviate_client_ready_ms=...` and `lifespan reranker_warmup_ms=...` (or reranker load lines from `retrieval.reranker`). Wrong or missing env should fail startup with `Missing required environment variables for API live mode`.

2. **`GET /health`** — `200`, body `{"status":"ok","service":"..."}`.

3. **`POST /query` happy path** — Short patient-voice question; `200`; `answer_markdown` with citations when retrieval hits; `sources` non-empty if synthesis cited; `X-Request-ID` response header.

4. **Weaviate connectivity** — If you get **empty** `sources` / empty answer for a query that works locally with the same cluster, re-check URL, API key, and collection. If you get **5xx** with `failed_stage` `searching` / `reading`, treat as retrieval/Weaviate path first.

5. **Warm latency (sanity, not a hard SLO)** — After one successful query (cold-ish), send **the same or similar query again**. Compare `metadata.latency_ms` (and stage breakdown) on the second response: total time should not be wildly higher than the first unless the platform throttled; if the **second** request is still extremely slow, check Railway CPU/RAM and logs for timeouts. Record rough numbers in your module log if you are tracking a ship gate.

6. **Log line** — In Railway logs, confirm a `query_stages` line for the happy path with `outcome=success` (matches automated log-shape expectations).
