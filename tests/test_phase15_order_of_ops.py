"""Phase 1.5 — policy strip must run before citation mapping (integration of helpers)."""
from __future__ import annotations

from src.api.citations import build_cited_sources
from src.api.policy_block import extract_policy_block
from src.retrieval.models import ChunkMetadata, SearchResult


def test_policy_strip_then_citations_on_cleaned_body() -> None:
    raw = (
        "Seek emergency care for crushing chest pain now. This tool cannot determine "
        "if it is only Long COVID.\n\n"
        "**Experiences**\n\nPeers mention urgency and follow-up [1]."
    )
    policy, cleaned = extract_policy_block(raw)
    assert policy.type == "emergency"
    assert "[1]" not in policy.markdown

    chunks = [
        SearchResult(
            chunk_id="c1",
            text="chunk one body",
            metadata=ChunkMetadata(
                chunk_id="c1",
                chunk_type="comment",
                permalink="/r/x/1",
                created_utc=1_700_000_000,
                comment_score=1,
            ),
        )
    ]
    sources, orphans = build_cited_sources(cleaned, chunks)
    assert orphans == []
    assert len(sources) == 1
    assert sources[0].n == 1
    assert sources[0].text == "chunk one body"
