"""
pipeline_cli.py — Dev CLI for end-to-end retrieve (+ optional synthesis).

Not production code. ``src.retrieval.pipeline.retrieve`` is the library entry
point for the FastAPI app; this module only wires CLI flags and printing.

Usage:
    cd projects/rag-longcovid-reddit-navigator
    python -m src.pipeline_cli --query "Has LDN helped anyone with Long COVID?"
    python -m src.pipeline_cli --query "..." --retrieve-only
    python -m src.pipeline_cli --query "..." -v

Environment:
    OPENROUTER_API_KEY — query rewrite (+ synthesis unless --retrieve-only)
    VOYAGE_API_KEY       — query embedding for hybrid search
    Weaviate             — same as hybrid_search / ranking_qa (e.g. WEAVIATE_URL)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.retrieval.config import RetrievalConfig
from src.retrieval.hybrid_search import _build_weaviate_client
from src.retrieval.pipeline import retrieve
from src.synthesis import SynthesisConfig, generate_synthesis


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dev harness: retrieve() → optional generate_synthesis()",
    )
    parser.add_argument("--query", "-q", type=str, required=True, help="Raw user query")
    parser.add_argument(
        "--retrieve-only",
        action="store_true",
        help="Run retrieval only (no OpenRouter call for synthesis)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="DEBUG logging for retrieval and synthesis",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)

    voyage_key = os.environ.get("VOYAGE_API_KEY", "").strip()
    if not voyage_key:
        raise SystemExit("VOYAGE_API_KEY is not set")

    if not os.environ.get("OPENROUTER_API_KEY", "").strip():
        raise SystemExit("OPENROUTER_API_KEY is not set (required for query rewrite)")

    retrieval_cfg = RetrievalConfig()
    retrieval_cfg.reranker.enabled = True

    client = _build_weaviate_client()
    try:
        print("\n" + "=" * 78)
        print("PIPELINE CLI — retrieve()" + (" + synthesis" if not args.retrieve_only else ""))
        print("=" * 78)
        print(f"\n[QUERY] {args.query!r}\n")

        retrieval = retrieve(
            args.query,
            cfg=retrieval_cfg,
            voyage_api_key=voyage_key,
            weaviate_client=client,
        )
        q = retrieval.query
        print(
            f"[REWRITE] mode={q.mode.value}  intent={q.intent.value}\n"
            f"  rewritten={q.best_rewrite.query!r}\n"
            f"  → {len(retrieval.results)} ranked chunks in {retrieval.elapsed_ms:.0f}ms\n"
        )

        if args.retrieve_only:
            print("[--retrieve-only] Top 5 chunk_id + preview:")
            for i, r in enumerate(retrieval.results[:5], start=1):
                prev = (r.text or "").replace("\n", " ")[:120]
                print(f"  [{i}] {r.chunk_id}  final={r.final_score:.4f}  {prev!r}")
            return

        cfg = SynthesisConfig()
        print(f"[SYNTHESIS] {cfg.model}")
        response = generate_synthesis(retrieval, cfg=cfg)

        print("\n" + "─" * 78)
        print("ANSWER")
        print("─" * 78)
        print(response.answer)
        print("─" * 78)
        print("\nSOURCES")
        print(json.dumps([s.model_dump() for s in response.sources], indent=2))
        print("\nMETADATA")
        print(response.metadata.model_dump_json(indent=2))

    finally:
        client.close()


if __name__ == "__main__":
    main()
