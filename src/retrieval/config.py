"""
config.py — Retrieval pipeline configuration.

All tunable parameters live here. Import RetrievalConfig and override fields
per environment or per request. Never hardcode values in pipeline modules.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from .models import FreshnessStance, IntentCategory


# ── Intent → freshness stance mapping ─────────────────────────────────────────

INTENT_FRESHNESS: dict[IntentCategory, FreshnessStance] = {
    IntentCategory.TREATMENT:  FreshnessStance.HIGH,    # guidance evolves — 2024 > 2020
    IntentCategory.TIMELINE:   FreshnessStance.MEDIUM,  # more data over time, but early stories valid
    IntentCategory.PREVALENCE: FreshnessStance.MEDIUM,  # community data evolves
    IntentCategory.SYMPTOM:    FreshnessStance.LOW,     # symptom descriptions are stable
    IntentCategory.EMOTIONAL:  FreshnessStance.LOW,     # lived experience is timeless
    IntentCategory.ADMIN:      FreshnessStance.LOW,     # rules/policy change but slowly
    IntentCategory.COMMUNITY:  FreshnessStance.LOW,     # community resources don't expire quickly
    IntentCategory.UNKNOWN:    FreshnessStance.MEDIUM,
}


# ── Query rewriter config ──────────────────────────────────────────────────────

class RewriterConfig(BaseModel):
    model: str = "google/gemini-2.5-flash-lite"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    temperature: float = 0.2          # low but not zero — slight variation helps with rewrites
    max_tokens: int = 512
    request_timeout_s: float = 30.0
    ambiguity_threshold: float = 0.75 # confidence below this → clarification mode


# ── Hybrid search config ───────────────────────────────────────────────────────

class SearchConfig(BaseModel):
    alpha: float = Field(default=0.8, ge=0.0, le=1.0,
                         description="BM25/vector ratio: 0=pure BM25, 1=pure vector")
    top_k_initial: int = Field(default=75, ge=1,
                               description="Candidates fetched from Weaviate — over-fetch to absorb dedup losses")
    top_k_deduped: int = Field(default=50, ge=1,
                               description="Target unique results after exact-text deduplication")
    voyage_embed_url: str = "https://api.voyageai.com/v1/embeddings"
    voyage_model: str = "voyage-4-large"
    voyage_timeout_s: float = 30.0


# ── Reranker config ────────────────────────────────────────────────────────────

class RerankerConfig(BaseModel):
    enabled: bool = False             # off by default; enable once core search is validated
    top_k_candidates: int = 50        # how many hybrid results to pass to the cross-encoder (matches top_k_deduped — ranking owns truncation)
    top_k_reranked: int = 50          # pass all deduped candidates to ranking layer; ranking owns truncation and diversity
    latency_budget_ms: float = 5000.0 # return original order if reranking exceeds this; 5s accommodates MPS cold-start (~2–3s) before warm inference (<500ms)


# ── Ranking config ─────────────────────────────────────────────────────────────

class RankingConfig(BaseModel):
    """
    Phase 3d ranking layer — signal weights and normalisation parameters.

    Weights are applied to normalised (0–1) signal values. The blended score
    is a weighted sum; weights do not need to sum to 1.0. Production defaults
    match Config B (rank-dominant with tiebreakers).

    Tournament configs (A/B/C/D) are defined in ranking.py for QA comparison.
    """
    w_rank_position: float = Field(default=1.0, ge=0.0,
        description="Reranker rank position — primary relevance signal")
    w_comment_score: float = Field(default=0.2, ge=0.0,
        description="Reddit upvotes on the chunk (comment_score or post_score)")
    w_num_comments: float = Field(default=0.1, ge=0.0,
        description="Parent post discussion volume")
    w_recency: float = Field(default=0.0, ge=0.0,
        description="Content recency (0 by default; enable for treatment queries)")
    w_agreement: float = Field(default=0.02, ge=0.0,
        description="Agreement presence flag — tiebreaker")
    w_thanks: float = Field(default=0.02, ge=0.0,
        description="Thanks presence flag — tiebreaker")

    diversity_cap: int = Field(default=3, ge=1,
        description="Max chunks per parent thread (link_id)")
    top_k_final: int = Field(default=25, ge=1,
        description="Final result count after ranking + truncation")

    comment_score_log_cap: float = Field(default=1500.0, gt=0,
        description="comment_score ceiling for log normalisation (corpus range: -100 to ~1500)")
    num_comments_log_cap: float = Field(default=2000.0, gt=0,
        description="num_comments ceiling for log normalisation")
    recency_half_life_days: float = Field(default=730.0, gt=0,
        description="Exponential decay half-life in days (default: 2 years)")


# ── Master config ──────────────────────────────────────────────────────────────

class RetrievalConfig(BaseModel):
    """
    Top-level config object passed through the retrieval pipeline.
    Override fields at call time; defaults are production-reasonable starting points.
    """
    rewriter: RewriterConfig = Field(default_factory=RewriterConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    ranking: RankingConfig = Field(default_factory=RankingConfig)

    # Convenience: intent → freshness lookup (not user-configurable)
    intent_freshness_map: dict[IntentCategory, FreshnessStance] = Field(
        default_factory=lambda: dict(INTENT_FRESHNESS)
    )

    def freshness_for(self, intent: IntentCategory) -> FreshnessStance:
        return self.intent_freshness_map.get(intent, FreshnessStance.MEDIUM)
