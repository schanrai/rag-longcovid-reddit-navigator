"""
models.py — Shared Pydantic types for the retrieval package.

All modules in src/retrieval/ import from here. No logic lives in this file.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────────────

class IntentCategory(str, Enum):
    SYMPTOM    = "symptom"     # what symptoms are, how they feel, identifying them
    TREATMENT  = "treatment"   # medications, therapies, interventions (incl. evidence questions)
    TIMELINE   = "timeline"    # recovery, progression, when things improve or worsen
    PREVALENCE = "prevalence"  # how common symptoms/experiences are in the community
    EMOTIONAL  = "emotional"   # coping, mental health, validation, grief, social support
    ADMIN      = "admin"       # insurance, disability, benefits, workplace, sick leave
    COMMUNITY  = "community"   # navigation, signposting resources, general Long COVID info
    UNKNOWN    = "unknown"     # cannot determine intent


class RewriteMode(str, Enum):
    CONFIDENT     = "confident"      # single rewrite, proceed directly
    CLARIFICATION = "clarification"  # ambiguous — present options to user


class RankingPreset(str, Enum):
    MOST_RELEVANT  = "most_relevant"   # intent-driven freshness defaults
    LATEST_FIRST   = "latest_first"    # recency dominates
    MOST_DISCUSSED = "most_discussed"  # num_comments primary, score secondary


class FreshnessStance(str, Enum):
    HIGH   = "high"    # newer posts ranked higher relative to popularity
    MEDIUM = "medium"  # balanced
    LOW    = "low"     # age barely affects ranking; popularity dominates


# ── Query rewriter ─────────────────────────────────────────────────────────────

class RewriteCandidate(BaseModel):
    """A single candidate interpretation of an ambiguous query."""
    query: str = Field(..., description="Rewritten query string")
    explanation: str = Field(..., description="Why this interpretation was chosen")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0–1")


class RewriteResult(BaseModel):
    """
    Output of query_rewriter.rewrite().

    mode=confident: rewrites contains exactly one candidate; proceed to retrieval.
    mode=clarification: rewrites contains 2–3 candidates; frontend presents options.
    Until frontend exists, the highest-confidence candidate is used as fallback.
    """
    mode: RewriteMode
    original_query: str
    rewrites: list[RewriteCandidate]
    intent: IntentCategory
    raw_llm_response: str = Field(default="", exclude=True)  # for debugging only

    @property
    def best_rewrite(self) -> RewriteCandidate:
        """Returns the highest-confidence rewrite candidate."""
        return max(self.rewrites, key=lambda r: r.confidence)


# ── Search / retrieval ─────────────────────────────────────────────────────────

class ChunkMetadata(BaseModel):
    """Metadata stored alongside each chunk in Weaviate."""
    chunk_id: str
    chunk_type: str                        # "comment" | "post"
    post_title: str = ""
    post_summary: str = ""
    permalink: str = ""
    created_utc: int | None = None
    comment_score: int | None = None
    post_score: int | None = None
    agreement_count: int = 0
    thanks_count: int = 0
    num_comments: int | None = None
    upvote_ratio: float | None = None
    nest_level: int | None = None
    is_submitter: bool = False
    stickied: bool = False
    chunk_index: int = 0
    total_chunks: int = 1
    word_count: int = 0
    link_flair_text: str | None = None
    link_id: str | None = None
    parent_id: str | None = None


class SearchResult(BaseModel):
    """A single retrieved chunk with its retrieval score and metadata."""
    chunk_id: str
    text: str
    hybrid_score: float = 0.0
    rerank_score: float | None = None
    final_score: float = 0.0            # set by ranking.py
    metadata: ChunkMetadata
    extra: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """Full output of the retrieval pipeline for one query."""
    query: RewriteResult
    results: list[SearchResult]
    preset: RankingPreset
    reranker_enabled: bool
    elapsed_ms: float = 0.0
