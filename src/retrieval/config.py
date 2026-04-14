"""
config.py — Retrieval pipeline configuration.

All tunable parameters live here. Import RetrievalConfig and override fields
per environment or per request. Never hardcode values in pipeline modules.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from .models import FreshnessStance, IntentCategory, RankingPreset


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
    top_k_candidates: int = 25        # how many hybrid results to pass to the cross-encoder
    top_k_reranked: int = 20          # how many reranked results to return
    latency_budget_ms: float = 2000.0 # return original order if reranking exceeds this


# ── Ranking config ─────────────────────────────────────────────────────────────

class RankingConfig(BaseModel):
    default_preset: RankingPreset = RankingPreset.MOST_RELEVANT

    # Popularity signal weights (applied before freshness adjustment)
    weight_comment_score: float = 1.0
    weight_post_score: float = 1.0
    weight_agreement_count: float = 1.5  # stronger signal — explicit community validation
    weight_thanks_count: float = 1.2
    weight_num_comments: float = 0.8

    # Freshness decay strengths per stance (implementation detail — tune during build)
    # These are relative multipliers; exact formula determined during 3d build.
    decay_high: float = 0.3    # strong time penalty on older posts
    decay_medium: float = 0.1  # moderate
    decay_low: float = 0.01    # near-zero — age barely matters

    # Preset overrides
    latest_first_decay_override: float = 0.5   # heavy recency dominance
    most_discussed_num_comments_weight: float = 3.0  # num_comments is primary


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
