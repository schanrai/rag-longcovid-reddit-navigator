#!/usr/bin/env python3
"""
eval_judge.py — Phase 1d: cosine top-10 retrieval + LLM-as-judge relevance scores.

Loads embeddings from embed_eval.py output, golden_queries.json for rubric
(notes + expected_terms), and eval_corpus.jsonl for chunk text.

Metrics (per model, per query, and mean by category):
  NDCG@10 — graded gains from judge scores {1,2,3}
  MRR@10  — reciprocal rank of first result with judge score >= 2

Usage:
  python3 src/eval_judge.py
  python3 src/eval_judge.py --models qwen3_embedding_0.6b --top-k 10
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import httpx
import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent))
from embed_eval import compose_embedding_input

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("eval_judge")

JUDGE_SYSTEM: Final[str] = (
    "You are a strict relevance judge for a Long COVID patient community search system. "
    "Reply with a single digit only: 1, 2, or 3."
)

JUDGE_USER_TEMPLATE: Final[str] = """You are evaluating search results for a Long COVID patient community RAG system.
Query: {query}
Relevance guidance: {notes}
Expected terms (hints only, not exhaustive): {expected_terms}
Result (full thread chunk as stored for retrieval):
{chunk_text}
Score 1 (not relevant), 2 (partially relevant), or 3 (highly relevant). Reply with just the number."""

SCORE_RE = re.compile(r"[123]")


def dcg_at_k(relevances: list[float], k: int) -> float:
    gains = np.asarray(relevances[:k], dtype=np.float64)
    if gains.size == 0:
        return 0.0
    positions = np.arange(2, gains.size + 2, dtype=np.float64)
    return float(np.sum(gains / np.log2(positions)))


def ndcg_at_k(relevances: list[float], k: int) -> float:
    dcg = dcg_at_k(relevances, k)
    ideal = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(relevances: list[int], k: int, *, min_relevant: int = 2) -> float:
    for i, rel in enumerate(relevances[:k]):
        if rel >= min_relevant:
            return 1.0 / (i + 1)
    return 0.0


def parse_judge_digit(content: str) -> int:
    m = SCORE_RE.search(content.strip())
    if not m:
        raise ValueError(f"No relevance digit in judge response: {content!r}")
    return int(m.group(0))


def load_golden_queries(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return list(raw["queries"])


def load_eval_corpus_ordered(path: Path) -> tuple[list[str], dict[str, str]]:
    """Line order must match embed_eval.py / index.json chunk_ids."""
    chunk_ids: list[str] = []
    texts: dict[str, str] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = rec.get("chunk_id")
            if not cid:
                continue
            cid = str(cid)
            chunk_ids.append(cid)
            texts[cid] = compose_embedding_input(rec)
    return chunk_ids, texts


def top_k_indices(query_vec: np.ndarray, chunk_mat: np.ndarray, k: int) -> list[int]:
    """Cosine similarity with L2-normalized rows."""
    sims = chunk_mat @ query_vec.reshape(-1)
    order = np.argsort(-sims)
    return [int(x) for x in order[:k]]


@dataclass(frozen=True)
class JudgeConfig:
    openrouter_base_url: str
    model: str
    max_concurrent: int
    retry_attempts: int
    timeout_s: float


def _openrouter_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/rag-longcovid-reddit-navigator",
        "X-Title": "Long COVID Reddit RAG eval_judge",
    }


async def judge_one(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    cfg: JudgeConfig,
    api_key: str,
    *,
    query: str,
    notes: str,
    expected_terms: str,
    chunk_text: str,
) -> int:
    user = JUDGE_USER_TEMPLATE.format(
        query=query,
        notes=notes,
        expected_terms=expected_terms,
        chunk_text=chunk_text,
    )
    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
    }
    url = f"{cfg.openrouter_base_url.rstrip('/')}/chat/completions"
    async with sem:
        last: Exception | None = None
        for attempt in range(cfg.retry_attempts):
            try:
                r = await client.post(
                    url,
                    headers=_openrouter_headers(api_key),
                    json=payload,
                )
                r.raise_for_status()
                body = r.json()
                choices = body.get("choices") or []
                if not choices:
                    raise ValueError("missing choices")
                content = (choices[0].get("message") or {}).get("content") or ""
                return parse_judge_digit(str(content))
            except Exception as exc:
                last = exc
                await asyncio.sleep(1.5 * (attempt + 1))
        assert last is not None
        raise last


class QueryEvalResult(BaseModel):
    query_id: str
    category: str
    top_chunk_ids: list[str] = Field(default_factory=list)
    judge_scores: list[int] = Field(default_factory=list)
    ndcg_at_k: float = 0.0
    mrr_at_k: float = 0.0


class ModelEvalReport(BaseModel):
    model_id: str
    top_k: int
    mean_ndcg: float
    mean_mrr: float
    by_category_ndcg: dict[str, float] = Field(default_factory=dict)
    by_category_mrr: dict[str, float] = Field(default_factory=dict)
    queries: list[QueryEvalResult] = Field(default_factory=list)


async def run_model_async(
    *,
    model_id: str,
    embeddings_dir: Path,
    queries: list[dict[str, Any]],
    chunk_ids: list[str],
    chunk_id_to_text: dict[str, str],
    top_k: int,
    judge_cfg: JudgeConfig,
    api_key: str,
) -> ModelEvalReport:
    index_path = embeddings_dir / model_id / "index.json"
    chunks_path = embeddings_dir / model_id / "chunks.npy"
    queries_path = embeddings_dir / model_id / "queries.npy"
    for p in (index_path, chunks_path, queries_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    idx_chunk_ids: list[str] = list(index["chunk_ids"])
    idx_query_ids: list[str] = list(index["query_ids"])
    if idx_chunk_ids != chunk_ids:
        raise ValueError(
            f"chunk_ids order mismatch between eval_corpus and {index_path}. "
            "Re-run embed_eval after build_eval_corpus."
        )

    chunk_mat = np.load(chunks_path)
    query_mat = np.load(queries_path)
    if query_mat.shape[0] != len(queries):
        raise ValueError("Query embedding count does not match golden_queries.json")

    qpos = {qid: i for i, qid in enumerate(idx_query_ids)}
    query_results: list[QueryEvalResult] = []

    sem = asyncio.Semaphore(judge_cfg.max_concurrent)
    timeout = httpx.Timeout(judge_cfg.timeout_s)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for qi, q in enumerate(queries):
            qid = str(q["id"])
            if qid not in qpos:
                raise ValueError(f"Query {qid} missing from embedding index")
            row = qpos[qid]
            qvec = query_mat[row]
            tidx = top_k_indices(qvec, chunk_mat, top_k)
            top_cids = [chunk_ids[i] for i in tidx]

            tasks = []
            for i in tidx:
                cid = chunk_ids[i]
                text = chunk_id_to_text.get(cid)
                if text is None:
                    raise KeyError(f"Chunk text missing for {cid}")
                tasks.append(
                    judge_one(
                        client,
                        sem,
                        judge_cfg,
                        api_key,
                        query=str(q.get("query") or ""),
                        notes=str(q.get("notes") or ""),
                        expected_terms=", ".join(q.get("expected_terms") or []),
                        chunk_text=text,
                    )
                )
            scores = await asyncio.gather(*tasks)
            nd = ndcg_at_k([float(s) for s in scores], top_k)
            mr = mrr_at_k(scores, top_k)
            query_results.append(
                QueryEvalResult(
                    query_id=qid,
                    category=str(q.get("category") or "unknown"),
                    top_chunk_ids=top_cids,
                    judge_scores=list(scores),
                    ndcg_at_k=nd,
                    mrr_at_k=mr,
                )
            )
            log.info("  %s %s NDCG@%s=%.4f MRR@%s=%.4f", qid, model_id, top_k, nd, top_k, mr)

    mean_ndcg = float(np.mean([q.ndcg_at_k for q in query_results])) if query_results else 0.0
    mean_mrr = float(np.mean([q.mrr_at_k for q in query_results])) if query_results else 0.0

    by_cat: dict[str, list[QueryEvalResult]] = {}
    for qr in query_results:
        by_cat.setdefault(qr.category, []).append(qr)

    by_ndcg = {c: float(np.mean([x.ndcg_at_k for x in lst])) for c, lst in by_cat.items()}
    by_mrr = {c: float(np.mean([x.mrr_at_k for x in lst])) for c, lst in by_cat.items()}

    return ModelEvalReport(
        model_id=model_id,
        top_k=top_k,
        mean_ndcg=mean_ndcg,
        mean_mrr=mean_mrr,
        by_category_ndcg=by_ndcg,
        by_category_mrr=by_mrr,
        queries=query_results,
    )


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="LLM-as-judge retrieval eval for embedding models.")
    ap.add_argument("--golden", type=Path, default=root / "data" / "golden_queries.json")
    ap.add_argument("--eval-corpus", type=Path, default=root / "data" / "eval_corpus.jsonl")
    ap.add_argument("--embeddings-dir", type=Path, default=root / "data" / "embeddings")
    ap.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Subdir names under embeddings/ (default: all that have index.json)",
    )
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--openrouter-model", type=str, default="google/gemini-2.5-flash-lite")
    ap.add_argument("--max-concurrent", type=int, default=6)
    ap.add_argument("--retry-attempts", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument(
        "--out",
        type=Path,
        default=root / "reports" / "embedding_eval_report.json",
    )
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        log.error("OPENROUTER_API_KEY not set")
        sys.exit(1)

    if not args.golden.exists() or not args.eval_corpus.exists():
        log.error("Missing golden queries or eval corpus")
        sys.exit(1)

    queries = load_golden_queries(args.golden)
    chunk_ids, chunk_id_to_text = load_eval_corpus_ordered(args.eval_corpus)

    emb_dir = args.embeddings_dir
    if args.models:
        model_ids = list(args.models)
    else:
        model_ids = sorted(
            p.name
            for p in emb_dir.iterdir()
            if p.is_dir() and (p / "index.json").exists()
        )
    if not model_ids:
        log.error("No embedding model directories found under %s", emb_dir)
        sys.exit(1)

    judge_cfg = JudgeConfig(
        openrouter_base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        model=args.openrouter_model,
        max_concurrent=args.max_concurrent,
        retry_attempts=args.retry_attempts,
        timeout_s=args.timeout,
    )

    reports: list[dict[str, Any]] = []
    for mid in model_ids:
        log.info("=== Judging model: %s ===", mid)
        try:
            rep = asyncio.run(
                run_model_async(
                    model_id=mid,
                    embeddings_dir=emb_dir,
                    queries=queries,
                    chunk_ids=chunk_ids,
                    chunk_id_to_text=chunk_id_to_text,
                    top_k=args.top_k,
                    judge_cfg=judge_cfg,
                    api_key=api_key,
                )
            )
            reports.append(json.loads(rep.model_dump_json()))
        except Exception as exc:
            log.exception("Model %s failed: %s", mid, exc)
            sys.exit(1)

    payload = {
        "schema": "embedding_eval_report_v1",
        "top_k": args.top_k,
        "judge_model": args.openrouter_model,
        "models": reports,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
