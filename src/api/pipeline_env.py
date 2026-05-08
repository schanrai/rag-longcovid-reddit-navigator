"""
Validate environment variables required for live API mode (Weaviate + LLM + embeddings).

Used at FastAPI startup when ``connect_weaviate`` is True so Railway fails fast
with a clear message instead of mid-request ``ValueError`` / ``EnvironmentError``.
"""
from __future__ import annotations

import os

_REQUIRED_FOR_LIVE: tuple[str, ...] = (
    "WEAVIATE_URL",
    "WEAVIATE_API_KEY",
    "OPENROUTER_API_KEY",
    "VOYAGE_API_KEY",
)


def validate_live_pipeline_env() -> None:
    """
    Ensure all keys needed for ``POST /query`` against Weaviate Cloud are present.

    Raises
    ------
    RuntimeError
        If any required variable is missing or blank.
    """
    missing = [name for name in _REQUIRED_FOR_LIVE if not os.environ.get(name, "").strip()]
    if missing:
        raise RuntimeError(
            "Missing required environment variables for API live mode: "
            + ", ".join(missing)
            + ". Set them in Railway or .env. For offline tests use "
            "create_app(connect_weaviate=False, warmup_reranker=False)."
        )
