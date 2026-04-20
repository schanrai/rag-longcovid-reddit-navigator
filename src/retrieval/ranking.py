"""
ranking.py — Phase 3d: Ranking layer (production).

Pipeline: 50 reranked candidates → diversity cap → score blending → truncate to 25.

Applies non-relevance signals (community validation, discussion volume, recency,
social tiebreakers) on top of the reranker's relevance ordering. All signal
values are normalised to 0–1 before blending.

Design decisions (see scope-v3.md Section 5.4 and reports/ranking_qa_round1_analysis.md):

- **Rank position, not raw rerank_score**: cross-encoder scores are unreliable
  across query types (emotional queries produce entirely negative scores for
  relevant content). Ordinal position avoids the calibration problem.
- **Diversity cap before blending**: prevents thread concentration from biasing
  the score blend. Cap operates on reranker ordering (highest relevance first).
  Position-based, not content-based — see analysis Section 7a for the assumption.
- **Social signals as presence flags**: agreement_count and thanks_count max
  at 4 and are >0 on only 0.9% / 4.1% of chunks. Fine-grained normalisation
  is meaningless — a binary flag preserves their tiebreaker role.
- **comment_score clamp-then-log**: negatives → 0 (downvoted content contributes
  nothing). Log-scale compresses the long tail correctly.
- **Recency exponential decay**: 2-year half-life. Default weight is 0.0 —
  only meaningful for treatment/research queries.

QA harness (tournament runner, single-query inspection, human-readable output)
lives in `src/ranking_qa.py`.
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict

from .config import RankingConfig
from .models import ChunkMetadata, SearchResult

log = logging.getLogger("retrieval.ranking")


# ── Normalisation functions ───────────────────────────────────────────────────

def norm_rank_position(rank: int, total: int) -> float:
    """Rank 1 (best) → 1.0, rank ``total`` (worst) → 0.0."""
    if total <= 1:
        return 1.0
    return (total - rank) / (total - 1)


def norm_comment_score(score: int | None, log_cap: float = 1500.0) -> float:
    """Clamp negatives to zero, then log-scale. Downvoted content → 0.0."""
    s = max(0, score or 0)
    if s <= 1:
        return 0.0
    return min(math.log(s) / math.log(log_cap), 1.0)


def norm_num_comments(count: int | None, log_cap: float = 2000.0) -> float:
    """Log-scale discussion volume. Zero or missing → 0.0."""
    n = max(1, count or 0)
    if n <= 1:
        return 0.0
    return min(math.log(n) / math.log(log_cap), 1.0)


def norm_recency(
    created_utc: int | None,
    reference_utc: int,
    half_life_days: float = 730.0,
) -> float:
    """Exponential decay. 2-year-old → 0.5, 4-year-old → 0.25."""
    if created_utc is None or reference_utc <= 0:
        return 0.0
    age_days = max(0, (reference_utc - created_utc) / 86400)
    return math.pow(2, -age_days / half_life_days)


def norm_social(count: int) -> float:
    """Presence flag: 1.0 if count > 0, else 0.0."""
    return 1.0 if count > 0 else 0.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chunk_community_score(m: ChunkMetadata) -> int:
    """
    Upvote-based community validation for any chunk type.

    Comment chunks use comment_score (their own upvotes); post chunks
    fall back to post_score since comment_score is typically absent.
    """
    if m.comment_score is not None:
        return m.comment_score
    if m.post_score is not None:
        return m.post_score
    return 0


# ── Diversity cap ─────────────────────────────────────────────────────────────

def apply_diversity_cap(
    results: list[SearchResult],
    cap: int,
) -> list[SearchResult]:
    """
    Limit chunks per parent thread (link_id). Keeps the first ``cap`` per
    thread in input order (reranker order — highest relevance first).
    Chunks without a link_id pass through unconditionally.
    """
    thread_counts: dict[str, int] = defaultdict(int)
    filtered: list[SearchResult] = []
    dropped = 0

    for r in results:
        lid = r.metadata.link_id
        if lid is None:
            filtered.append(r)
            continue
        if thread_counts[lid] < cap:
            thread_counts[lid] += 1
            filtered.append(r)
        else:
            dropped += 1

    if dropped:
        log.info(
            "Diversity cap (%d/thread) removed %d chunks — %d remain",
            cap, dropped, len(filtered),
        )
    return filtered


# ── Score blending ────────────────────────────────────────────────────────────

def rank(
    results: list[SearchResult],
    *,
    cfg: RankingConfig | None = None,
) -> list[SearchResult]:
    """
    Phase 3d ranking: diversity cap → score blending → truncation.

    Parameters
    ----------
    results:
        Reranked candidates (typically 50), sorted by rerank_score descending.
    cfg:
        RankingConfig with signal weights and normalisation parameters.

    Returns
    -------
    list[SearchResult] sorted by blended final_score, truncated to top_k_final.
    """
    if cfg is None:
        cfg = RankingConfig()

    if not results:
        return []

    total = len(results)

    original_ranks = {r.chunk_id: i for i, r in enumerate(results, 1)}

    reference_utc = max(
        (r.metadata.created_utc for r in results if r.metadata.created_utc is not None),
        default=0,
    )

    diverse = apply_diversity_cap(results, cfg.diversity_cap)

    ranked: list[SearchResult] = []
    for r in diverse:
        m = r.metadata
        rerank_rank = original_ranks[r.chunk_id]
        community_score = _chunk_community_score(m)

        n_rank = norm_rank_position(rerank_rank, total)
        n_comment = norm_comment_score(community_score, cfg.comment_score_log_cap)
        n_num_comm = norm_num_comments(m.num_comments, cfg.num_comments_log_cap)
        n_recency = norm_recency(m.created_utc, reference_utc, cfg.recency_half_life_days)
        n_agree = norm_social(m.agreement_count)
        n_thanks = norm_social(m.thanks_count)

        blended = (
            cfg.w_rank_position * n_rank
            + cfg.w_comment_score * n_comment
            + cfg.w_num_comments * n_num_comm
            + cfg.w_recency * n_recency
            + cfg.w_agreement * n_agree
            + cfg.w_thanks * n_thanks
        )

        updated = r.model_copy(update={
            "final_score": blended,
            "extra": {
                **r.extra,
                "rerank_rank": rerank_rank,
                "community_score_raw": community_score,
                "n_rank_position": round(n_rank, 4),
                "n_comment_score": round(n_comment, 4),
                "n_num_comments": round(n_num_comm, 4),
                "n_recency": round(n_recency, 4),
                "n_agreement": round(n_agree, 4),
                "n_thanks": round(n_thanks, 4),
            },
        })
        ranked.append(updated)

    ranked.sort(key=lambda r: r.final_score, reverse=True)
    final = ranked[: cfg.top_k_final]

    log.info(
        "Ranking: %d → %d (diversity cap) → %d (truncated)",
        total, len(diverse), len(final),
    )
    return final
