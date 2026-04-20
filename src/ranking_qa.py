"""
ranking_qa.py — Phase 3d ranking weight tournament & single-query QA harness.

Not production code. This module provides the tournament runner, single-query
runner, and human-readable output formatting used to evaluate ranking signal
weight blends. Production ranking logic lives in `src/retrieval/ranking.py`.

The Round 1 tournament (April 17, 2026) locked Config B as the production
default. This harness remains available for Round 2 (if ever needed) and for
ad-hoc single-query inspection during ongoing development.

CLI usage:
    python -m src.ranking_qa --query "..." --config B_rank_dominant
    python -m src.ranking_qa --tournament
    python -m src.ranking_qa --tournament --output-dir reports/ranking_qa_round2/

See `reports/ranking_qa_round1_analysis.md` for Round 1 methodology and verdict.
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

from src.retrieval.config import RankingConfig, RetrievalConfig
from src.retrieval.hybrid_search import _build_weaviate_client, search as hybrid_search
from src.retrieval.models import SearchResult
from src.retrieval.ranking import rank
from src.retrieval.reranker import rerank

load_dotenv()
log = logging.getLogger("ranking_qa")


# ── Tournament definitions ────────────────────────────────────────────────────

TOURNAMENT_QUERIES: Final[dict[str, str]] = {
    "Q1": "I miss my life",
    "Q2": "Anyone taking beta blockers for tachycardia-like symptoms?",
    "Q3": "Has LDN helped anyone with Long COVID symptoms?",
    "Q4": "why am i still so exhausted 8 months after getting Covid? Is this normal??",
    "Q5": "Finally recovered after 4 years - here are the treatments I tried",
}

TOURNAMENT_CONFIGS: Final[dict[str, RankingConfig]] = {
    "A_rank_only": RankingConfig(
        w_rank_position=1.0, w_comment_score=0.0, w_num_comments=0.0,
        w_recency=0.0, w_agreement=0.0, w_thanks=0.0,
    ),
    "B_rank_dominant": RankingConfig(
        w_rank_position=1.0, w_comment_score=0.2, w_num_comments=0.1,
        w_recency=0.0, w_agreement=0.02, w_thanks=0.02,
    ),
    "C_community_dominant": RankingConfig(
        w_rank_position=0.4, w_comment_score=0.4, w_num_comments=0.3,
        w_recency=0.0, w_agreement=0.05, w_thanks=0.05,
    ),
    "D_recency_treatment": RankingConfig(
        w_rank_position=1.0, w_comment_score=0.2, w_num_comments=0.1,
        w_recency=0.15, w_agreement=0.02, w_thanks=0.02,
    ),
}

CONFIG_D_QUERIES: Final[set[str]] = {"Q2", "Q3", "Q5"}


# ── Output formatting ─────────────────────────────────────────────────────────

def _format_config_header(name: str, cfg: RankingConfig) -> str:
    return (
        f"Config: {name}\n"
        f"  w_rank={cfg.w_rank_position}  w_cscore={cfg.w_comment_score}  "
        f"w_ncomm={cfg.w_num_comments}  w_recency={cfg.w_recency}  "
        f"w_agree={cfg.w_agreement}  w_thanks={cfg.w_thanks}\n"
        f"  diversity_cap={cfg.diversity_cap}  top_k_final={cfg.top_k_final}"
    )


def _format_result_row(final_rank: int, r: SearchResult) -> str:
    m = r.metadata
    ex = r.extra
    rerank_rank = ex.get("rerank_rank", "?")
    rerank_str = f"{r.rerank_score:.4f}" if r.rerank_score is not None else "n/a"
    community_raw = ex.get("community_score_raw", "?")

    age_str = ""
    if m.created_utc:
        age_days = (time.time() - m.created_utc) / 86400
        age_str = f"{age_days:.0f}d"

    if isinstance(rerank_rank, int):
        movement = rerank_rank - final_rank
        move_str = f"↑{movement}" if movement > 0 else (f"↓{abs(movement)}" if movement < 0 else "—")
    else:
        move_str = "?"

    preview = r.text.replace("\n", " ")[:200]

    line1 = (
        f"  [{final_rank:02d}] final={r.final_score:.4f}  "
        f"rerank_pos={rerank_rank:<3}  move={move_str:<5}  "
        f"rerank_raw={rerank_str}"
    )
    line2 = (
        f"       cscore={community_raw:<6}  ncomm={m.num_comments or 0:<5}  "
        f"agree={m.agreement_count}  thanks={m.thanks_count}  "
        f"age={age_str:<6}  link_id={m.link_id or 'n/a'}"
    )
    line3 = (
        f"       n_rank={ex.get('n_rank_position', '?'):<7}  "
        f"n_cscore={ex.get('n_comment_score', '?'):<7}  "
        f"n_ncomm={ex.get('n_num_comments', '?'):<7}  "
        f"n_recency={ex.get('n_recency', '?'):<7}  "
        f"n_agree={ex.get('n_agreement', '?'):<5}  "
        f"n_thanks={ex.get('n_thanks', '?')}"
    )
    line4 = f"       {preview!r}"

    return f"{line1}\n{line2}\n{line3}\n{line4}"


def _format_query_block(
    query_id: str,
    query: str,
    config_name: str,
    cfg: RankingConfig,
    results: list[SearchResult],
    top_n: int = 25,
) -> str:
    lines = [
        f"\n{'─' * 78}",
        f"Query [{query_id}]: {query!r}",
        _format_config_header(config_name, cfg),
        f"{'─' * 78}",
    ]

    display = results[:top_n]
    # Post chunks have link_id=None; fall back to chunk_id so each post counts
    # as its own distinct "thread" in the diversity display.
    thread_keys_10 = [r.metadata.link_id or r.chunk_id for r in display[:10]]
    thread_keys_all = [r.metadata.link_id or r.chunk_id for r in display]
    lines.append(
        f"  Thread diversity — top 10: {len(set(thread_keys_10))} unique  |  "
        f"top {len(display)}: {len(set(thread_keys_all))} unique"
    )
    lines.append("")

    for i, r in enumerate(display, 1):
        lines.append(_format_result_row(i, r))
        if i == 10 and len(display) > 10:
            lines.append(f"  {'· ' * 39}")

    return "\n".join(lines)


# ── Tournament runner ─────────────────────────────────────────────────────────

def run_tournament(output_dir: Path | None = None) -> None:
    """
    Run the full ranking tournament: 5 queries × 3 universal configs + 3
    treatment queries × Config D = 18 runs.

    Retrieval + reranking runs once per query; ranking is applied per config
    on the cached candidates to avoid redundant API calls.
    """
    retrieval_cfg = RetrievalConfig()
    retrieval_cfg.reranker.enabled = True

    voyage_key = os.environ.get("VOYAGE_API_KEY", "")
    if not voyage_key:
        raise EnvironmentError("VOYAGE_API_KEY not set")

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    total_runs = sum(
        1
        for cname in TOURNAMENT_CONFIGS
        for qid in TOURNAMENT_QUERIES
        if cname != "D_recency_treatment" or qid in CONFIG_D_QUERIES
    )

    print("=" * 78)
    print("RANKING TOURNAMENT — Round 1")
    print(f"Queries: {len(TOURNAMENT_QUERIES)}  |  Configs: {len(TOURNAMENT_CONFIGS)}  |  Total runs: {total_runs}")
    print(f"Generated: {datetime.now().isoformat()}")
    print("=" * 78)

    client = _build_weaviate_client()
    try:
        reranked_cache: dict[str, list[SearchResult]] = {}
        for qid, query in TOURNAMENT_QUERIES.items():
            print(f"\n[RETRIEVE+RERANK] {qid}: {query!r}")
            t0 = time.perf_counter()
            hybrid_results = hybrid_search(
                query, cfg=retrieval_cfg, voyage_api_key=voyage_key, weaviate_client=client,
            )
            reranked = rerank(query, hybrid_results, cfg=retrieval_cfg)
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"  → {len(reranked)} reranked candidates in {elapsed:.0f}ms")
            reranked_cache[qid] = reranked

        per_query_output: dict[str, list[str]] = defaultdict(list)

        for config_name, ranking_cfg in TOURNAMENT_CONFIGS.items():
            for qid, query in TOURNAMENT_QUERIES.items():
                if config_name == "D_recency_treatment" and qid not in CONFIG_D_QUERIES:
                    continue

                ranked = rank(reranked_cache[qid], cfg=ranking_cfg)
                block = _format_query_block(qid, query, config_name, ranking_cfg, ranked)
                print(block)
                per_query_output[qid].append(block)

        if output_dir:
            for qid in TOURNAMENT_QUERIES:
                filepath = output_dir / f"{qid}_comparison.txt"
                with open(filepath, "w") as f:
                    f.write(f"RANKING TOURNAMENT — Round 1\n")
                    f.write(f"Query [{qid}]: {TOURNAMENT_QUERIES[qid]!r}\n")
                    f.write(f"Generated: {datetime.now().isoformat()}\n")
                    f.write("=" * 78 + "\n")
                    for block in per_query_output[qid]:
                        f.write(block)
                        f.write("\n\n")
                print(f"  → Saved {filepath}")

    finally:
        client.close()

    print("\n" + "=" * 78)
    print("Tournament complete.")
    if output_dir:
        print(f"Output files: {output_dir}/")
    print("=" * 78)


# ── Single query runner ───────────────────────────────────────────────────────

def run_single(query: str, config_name: str, top_n: int = 25) -> None:
    if config_name not in TOURNAMENT_CONFIGS:
        available = ", ".join(TOURNAMENT_CONFIGS.keys())
        raise ValueError(f"Unknown config {config_name!r}. Available: {available}")

    ranking_cfg = TOURNAMENT_CONFIGS[config_name]
    retrieval_cfg = RetrievalConfig()
    retrieval_cfg.reranker.enabled = True

    voyage_key = os.environ.get("VOYAGE_API_KEY", "")
    if not voyage_key:
        raise EnvironmentError("VOYAGE_API_KEY not set")

    print(f"\n[RETRIEVE+RERANK] {query!r}")
    t0 = time.perf_counter()
    hybrid_results = hybrid_search(query, cfg=retrieval_cfg, voyage_api_key=voyage_key)
    reranked = rerank(query, hybrid_results, cfg=retrieval_cfg)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  → {len(reranked)} reranked in {elapsed:.0f}ms")

    ranked = rank(reranked, cfg=ranking_cfg)
    print(_format_query_block("ad-hoc", query, config_name, ranking_cfg, ranked, top_n=top_n))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Phase 3d ranking weight tournament & single-query QA")
    parser.add_argument("--query", type=str, help="Run a single query")
    parser.add_argument(
        "--config", type=str, default="B_rank_dominant",
        help=f"Config name. Options: {', '.join(TOURNAMENT_CONFIGS.keys())}",
    )
    parser.add_argument("--tournament", action="store_true", help="Run full ranking tournament")
    parser.add_argument("--output-dir", type=str, default=None, help="Save per-query comparison files")
    parser.add_argument("--top-n", type=int, default=25, help="Results to display (default 25)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger("retrieval").setLevel(logging.DEBUG)
        logging.getLogger("ranking_qa").setLevel(logging.DEBUG)

    if args.tournament:
        out = Path(args.output_dir) if args.output_dir else None
        run_tournament(output_dir=out)
    elif args.query:
        run_single(args.query, args.config, top_n=args.top_n)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
