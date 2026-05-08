"""Unit tests for cited-only source construction."""
from __future__ import annotations

from src.api.citations import build_cited_sources, ordered_citation_indices
from src.retrieval.models import ChunkMetadata, SearchResult


def _chunk(i: int, text: str = "body") -> SearchResult:
    return SearchResult(
        chunk_id=f"id{i}",
        text=text,
        metadata=ChunkMetadata(
            chunk_id=f"id{i}",
            chunk_type="comment",
            permalink=f"/p{i}",
            created_utc=1700000000 + i,
            comment_score=i,
        ),
    )


def test_ordered_citation_indices_dedupe_and_order() -> None:
    assert ordered_citation_indices("A [3] B [1] C [3] D [2]") == [3, 1, 2]


def test_orphan_anchor_dropped() -> None:
    chunks = [_chunk(1), _chunk(2)]
    sources, orphans = build_cited_sources("See [1] and [99].", chunks)
    assert [s.n for s in sources] == [1]
    assert orphans == [99]


def test_sources_sorted_by_n() -> None:
    chunks = [_chunk(1), _chunk(2), _chunk(3)]
    sources, _ = build_cited_sources("Second [2] first [1].", chunks)
    assert [s.n for s in sources] == [1, 2]


def test_non_cited_chunk_not_in_sources() -> None:
    chunks = [_chunk(1, "one"), _chunk(2, "two"), _chunk(3, "three")]
    sources, _ = build_cited_sources("Only [1] and [3].", chunks)
    assert [s.n for s in sources] == [1, 3]
    assert {s.text for s in sources} == {"one", "three"}
    assert all(s.n != 2 for s in sources)
