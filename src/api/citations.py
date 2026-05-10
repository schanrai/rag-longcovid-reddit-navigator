"""
Cited-only ``sources`` for POST /query (Phase 6 Q2) — parse ``[n]`` after policy strip.
"""
from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from typing import Final

from pydantic import BaseModel, Field

from src.retrieval.models import SearchResult

log = logging.getLogger(__name__)

_ANCHOR: Final[re.Pattern[str]] = re.compile(r"\[(\d+)\]")


class CitedSource(BaseModel):
    """One cited chunk for the Links tab / sidebar."""

    n: int = Field(..., ge=1, description="Citation anchor index")
    text: str
    post_title: str = Field(
        default="",
        description="Parent post title from chunk metadata (Weaviate post_title)",
    )
    chunk_type: str
    comment_score: int | None = None
    post_score: int | None = None
    num_comments: int | None = None
    post_summary: str | None = None
    created_utc: str | None = Field(
        default=None,
        description="ISO-8601 UTC timestamp when available",
    )
    permalink: str


def ordered_citation_indices(answer_markdown: str) -> list[int]:
    """Unique ``[n]`` anchors in order of first appearance."""
    seen: set[int] = set()
    ordered: list[int] = []
    for m in _ANCHOR.finditer(answer_markdown or ""):
        n = int(m.group(1))
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    return ordered


def _iso_utc(ts: int | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(int(ts), tz=UTC).isoformat()


def build_cited_sources(
    answer_markdown: str,
    ranked_chunks: list[SearchResult],
) -> tuple[list[CitedSource], list[int]]:
    """
    Map citation anchors in ``answer_markdown`` to retrieved chunks.

    Returns ``(sources, orphan_indices)`` where orphans are ``[n]`` out of range
    (logged by caller).
    """
    indices = ordered_citation_indices(answer_markdown)
    n_chunks = len(ranked_chunks)
    sources: list[CitedSource] = []
    orphans: list[int] = []

    for n in sorted(indices):
        if n < 1 or n > n_chunks:
            orphans.append(n)
            continue
        r = ranked_chunks[n - 1]
        m = r.metadata
        sources.append(
            CitedSource(
                n=n,
                text=r.text or "",
                post_title=(m.post_title or "").strip(),
                chunk_type=m.chunk_type or "",
                comment_score=m.comment_score,
                post_score=m.post_score,
                num_comments=m.num_comments,
                post_summary=(m.post_summary or None) if (m.post_summary or "").strip() else None,
                created_utc=_iso_utc(m.created_utc),
                permalink=m.permalink or "",
            )
        )

    if orphans:
        log.warning(
            "citation orphans (no chunk for anchor): %s (chunk_count=%d)",
            orphans,
            n_chunks,
        )

    return sources, orphans
