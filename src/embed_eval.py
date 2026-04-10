#!/usr/bin/env python3
"""
embed_eval.py — Phase 1c: embed eval corpus + golden queries with each shortlisted model.

Comment chunks: post_title + post_summary (if present) + text.
Post chunks: post_title + text.

Self-hosted models via sentence-transformers (CPU). Voyage via httpx REST API.

Usage:
  python3 src/embed_eval.py
  python3 src/embed_eval.py --models qwen3_embedding_0.6b voyage_4_large
  python3 src/embed_eval.py --skip-voyage   # local models only
  python3 src/embed_eval.py --st-batch-size 8   # lower RAM (Rosetta / long chunks)

Long runs: see docs/mac-long-jobs-and-logs.md (nohup, log file, Console.app).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import httpx
import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("embed_eval")

SCHEMA_INDEX: Final[str] = "embedding_eval_index_v1"
VOYAGE_EMBED_URL: Final[str] = "https://api.voyageai.com/v1/embeddings"
VOYAGE_BATCH_MAX: Final[int] = 64


class EmbeddingModelSpec(BaseModel):
    """One eval embedding backend."""

    id: str = Field(description="Filesystem-safe directory name under data/embeddings/")
    kind: Literal["sentence_transformers", "voyage"]
    sentence_transformers_model: str | None = None
    voyage_model: str | None = None


DEFAULT_MODELS: list[EmbeddingModelSpec] = [
    EmbeddingModelSpec(
        id="qwen3_embedding_0.6b",
        kind="sentence_transformers",
        sentence_transformers_model="Qwen/Qwen3-Embedding-0.6B",
    ),
    EmbeddingModelSpec(
        id="jasper_token_compression_600m",
        kind="sentence_transformers",
        sentence_transformers_model="infgrad/Jasper-Token-Compression-600M",
    ),
    EmbeddingModelSpec(
        id="qwen3_embedding_4b",
        kind="sentence_transformers",
        sentence_transformers_model="Qwen/Qwen3-Embedding-4B",
    ),
    EmbeddingModelSpec(
        id="voyage_4_large",
        kind="voyage",
        voyage_model="voyage-4-large",
    ),
]


def compose_embedding_input(rec: dict[str, Any]) -> str:
    """Match scope: comment vs post composition."""
    title = (rec.get("post_title") or "").strip()
    body = (rec.get("text") or "").strip()
    cid = str(rec.get("chunk_id") or "")
    parts: list[str]
    if cid.startswith("t1_"):
        summary = rec.get("post_summary")
        if summary is not None and str(summary).strip():
            parts = [title, str(summary).strip(), body]
        else:
            parts = [title, body]
    else:
        parts = [title, body]
    return "\n\n".join(p for p in parts if p)


def load_golden_queries(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return list(raw["queries"])


def load_eval_corpus(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    chunk_ids: list[str] = []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = rec.get("chunk_id")
            if not cid:
                raise ValueError("eval_corpus line missing chunk_id")
            chunk_ids.append(str(cid))
            records.append(rec)
    return chunk_ids, records


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return mat / norms


def _encode_st(
    model: Any,
    texts: list[str],
    batch_size: int,
    desc: str,
) -> np.ndarray:
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    if not isinstance(vecs, np.ndarray):
        vecs = np.asarray(vecs, dtype=np.float32)
    log.info("  %s: shape %s", desc, vecs.shape)
    return vecs.astype(np.float32, copy=False)


def embed_sentence_transformers_both(
    *,
    model_name: str,
    chunk_texts: list[str],
    query_texts: list[str],
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    from sentence_transformers import SentenceTransformer

    log.info("Loading SentenceTransformer %s …", model_name)
    model = SentenceTransformer(
        model_name,
        trust_remote_code=True,
        device=device,
    )
    chunk_vecs = _encode_st(model, chunk_texts, batch_size, "chunks")
    query_vecs = _encode_st(model, query_texts, batch_size, "queries")
    return chunk_vecs, query_vecs


def voyage_embed_all(
    *,
    api_key: str,
    model: str,
    texts: list[str],
    input_type: str,
    batch_size: int,
) -> np.ndarray:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    out: list[list[float]] = []
    n = len(texts)
    with httpx.Client(timeout=120.0) as client:
        for start in range(0, n, batch_size):
            batch = texts[start : start + batch_size]
            payload: dict[str, Any] = {
                "model": model,
                "input": batch,
                "input_type": input_type,
            }
            for attempt in range(5):
                try:
                    r = client.post(VOYAGE_EMBED_URL, headers=headers, json=payload)
                    r.raise_for_status()
                    body = r.json()
                    data = body.get("data")
                    if not data or len(data) != len(batch):
                        raise ValueError("Voyage response data length mismatch")
                    # API returns sorted by index
                    data_sorted = sorted(data, key=lambda x: int(x.get("index", 0)))
                    for row in data_sorted:
                        emb = row.get("embedding")
                        if not emb:
                            raise ValueError("Missing embedding in Voyage row")
                        out.append(emb)
                    log.info("  Voyage embedded %s/%s", min(start + batch_size, n), n)
                    break
                except (httpx.HTTPError, ValueError) as exc:
                    wait = 2.0 * (attempt + 1)
                    log.warning("Voyage batch %s failed (%s), retry in %ss", start, exc, wait)
                    time.sleep(wait)
            else:
                raise RuntimeError(f"Voyage embedding failed at offset {start}") from None
    arr = np.asarray(out, dtype=np.float32)
    return _l2_normalize(arr)


@dataclass(frozen=True)
class RunPaths:
    root: Path
    golden_path: Path
    eval_corpus_path: Path
    embeddings_dir: Path


def run_one_model(
    spec: EmbeddingModelSpec,
    *,
    chunk_texts: list[str],
    query_texts: list[str],
    query_ids: list[str],
    chunk_ids: list[str],
    paths: RunPaths,
    st_batch_size: int,
    st_device: str,
    voyage_batch_size: int,
) -> None:
    out_dir = paths.embeddings_dir / spec.id
    out_dir.mkdir(parents=True, exist_ok=True)

    if spec.kind == "sentence_transformers":
        assert spec.sentence_transformers_model
        chunk_vecs, query_vecs = embed_sentence_transformers_both(
            model_name=spec.sentence_transformers_model,
            chunk_texts=chunk_texts,
            query_texts=query_texts,
            batch_size=st_batch_size,
            device=st_device,
        )
    else:
        key = os.environ.get("VOYAGE_API_KEY", "").strip()
        if not key:
            raise ValueError("VOYAGE_API_KEY not set in environment")
        assert spec.voyage_model
        chunk_vecs = voyage_embed_all(
            api_key=key,
            model=spec.voyage_model,
            texts=chunk_texts,
            input_type="document",
            batch_size=voyage_batch_size,
        )
        query_vecs = voyage_embed_all(
            api_key=key,
            model=spec.voyage_model,
            texts=query_texts,
            input_type="query",
            batch_size=min(voyage_batch_size, len(query_texts)),
        )

    dim = int(chunk_vecs.shape[1])
    np.save(out_dir / "chunks.npy", chunk_vecs)
    np.save(out_dir / "queries.npy", query_vecs)

    index = {
        "schema": SCHEMA_INDEX,
        "model_spec_id": spec.id,
        "kind": spec.kind,
        "sentence_transformers_model": spec.sentence_transformers_model,
        "voyage_model": spec.voyage_model,
        "chunk_ids": chunk_ids,
        "query_ids": query_ids,
        "dim": dim,
        "chunks_count": len(chunk_ids),
        "queries_count": len(query_ids),
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    log.info("Wrote %s (dim=%s)", out_dir, dim)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="Embed eval corpus + golden queries per model.")
    ap.add_argument("--golden", type=Path, default=root / "data" / "golden_queries.json")
    ap.add_argument("--eval-corpus", type=Path, default=root / "data" / "eval_corpus.jsonl")
    ap.add_argument("--out-dir", type=Path, default=root / "data" / "embeddings")
    ap.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Model spec ids (default: all). Example: qwen3_embedding_0.6b voyage_4_large",
    )
    ap.add_argument("--skip-voyage", action="store_true")
    ap.add_argument("--st-batch-size", type=int, default=32)
    ap.add_argument("--st-device", type=str, default="cpu")
    ap.add_argument("--voyage-batch-size", type=int, default=VOYAGE_BATCH_MAX)
    args = ap.parse_args()

    if not args.eval_corpus.exists():
        log.error("Missing eval corpus: %s — run build_eval_corpus.py first", args.eval_corpus)
        sys.exit(1)
    if not args.golden.exists():
        log.error("Missing golden queries: %s", args.golden)
        sys.exit(1)

    chunk_ids, records = load_eval_corpus(args.eval_corpus)
    chunk_texts = [compose_embedding_input(r) for r in records]

    queries = load_golden_queries(args.golden)
    query_ids = [str(q["id"]) for q in queries]
    query_texts = [str(q.get("query") or "").strip() for q in queries]

    specs = DEFAULT_MODELS
    if args.skip_voyage:
        specs = [s for s in specs if s.kind != "voyage"]
    if args.models:
        want = set(args.models)
        specs = [s for s in specs if s.id in want]
        missing = want - {s.id for s in specs}
        if missing:
            log.error("Unknown model ids: %s", sorted(missing))
            sys.exit(1)

    paths = RunPaths(
        root=root,
        golden_path=args.golden,
        eval_corpus_path=args.eval_corpus,
        embeddings_dir=args.out_dir,
    )

    for spec in specs:
        log.info("=== Embedding model: %s ===", spec.id)
        try:
            run_one_model(
                spec,
                chunk_texts=chunk_texts,
                query_texts=query_texts,
                query_ids=query_ids,
                chunk_ids=chunk_ids,
                paths=paths,
                st_batch_size=args.st_batch_size,
                st_device=args.st_device,
                voyage_batch_size=args.voyage_batch_size,
            )
        except Exception as exc:
            log.exception("Failed model %s: %s", spec.id, exc)
            sys.exit(1)


if __name__ == "__main__":
    main()
