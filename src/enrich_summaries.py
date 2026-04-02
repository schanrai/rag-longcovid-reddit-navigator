#!/usr/bin/env python3
"""
enrich_summaries.py — LLM post summaries for comment chunk context (v1)

Pass A — For each indexed post (same filter as chunk_data.py post chunking), call
         OpenRouter once to produce a 1–2 sentence summary. Checkpoint to
         data/post_summaries.json (resumable).
Pass B — Stream data/comment_chunks.jsonl and fill post_summary from the lookup
         by bare post id (link_id[3:]). Writes data/comment_chunks_enriched.jsonl.

Post summaries attach only to comment chunks (scope Section 6.3).

Usage:
  cd projects/rag-longcovid-reddit-navigator && python3 src/enrich_summaries.py
  python3 src/enrich_summaries.py --pass1-only
  python3 src/enrich_summaries.py --pass2-only
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from chunk_data import post_content_exclusion
from gate_analysis import stream_ndjson

load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("enrich_summaries")

SYSTEM_PROMPT = (
    "You are summarizing Reddit posts from r/LongCovid for a medical information "
    "retrieval system. Your summaries will be embedded alongside community comments "
    "so that short comments remain topically grounded."
)

USER_PROMPT_TEMPLATE = """Summarize the following post in 1-2 sentences. Focus on: the specific symptoms, conditions, or treatments mentioned, and the question or experience being shared. Use the same informal language patients use (e.g. "brain fog" not "cognitive impairment" unless the post uses clinical terms). Be specific — avoid vague summaries like "user asks about Long COVID".

Title: {title}

Post: {selftext}"""


@dataclass(frozen=True)
class Config:
    data_dir: Path = Path(__file__).parent.parent / "data"
    reports_dir: Path = Path(__file__).parent.parent / "reports"
    posts_file: str = "r_LongCovid_posts.jsonl"
    summaries_cache: str = "post_summaries.json"
    comment_chunks_in: str = "comment_chunks.jsonl"
    comment_chunks_out: str = "comment_chunks_enriched.jsonl"
    report_out: str = "enrich_report.json"

    model: str = "google/gemini-2.5-flash-lite"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    max_concurrent: int = 20
    retry_attempts: int = 3
    retry_delay_s: float = 2.0
    request_timeout_s: float = 120.0
    checkpoint_batch_size: int = 100

    progress_interval: int = 500


@dataclass
class Pass1Stats:
    total_indexed_posts: int = 0
    already_cached: int = 0
    api_calls: int = 0
    successes: int = 0
    failures: int = 0


@dataclass
class Pass2Stats:
    total_chunks: int = 0
    with_summary: int = 0
    missing_lookup: int = 0


def _openrouter_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/rag-longcovid-reddit-navigator",
        "X-Title": "Long COVID Reddit RAG enrich_summaries",
    }


def load_indexed_posts(cfg: Config) -> list[dict[str, Any]]:
    """Posts that pass post_content_exclusion (same set as post chunking)."""
    posts_path = cfg.data_dir / cfg.posts_file
    out: list[dict[str, Any]] = []
    for _, record, error in stream_ndjson(posts_path):
        if error or record is None:
            continue
        if post_content_exclusion(record) is not None:
            continue
        post_id = record.get("id")
        if not post_id:
            continue
        out.append(
            {
                "id": post_id,
                "title": (record.get("title") or "").strip(),
                "selftext": (record.get("selftext") or "").strip(),
            }
        )
    return out


def load_checkpoint(path: Path) -> dict[str, str | None]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    # Normalize: values may be str or null
    return {k: (v if v is None else str(v)) for k, v in raw.items()}


def save_checkpoint(path: Path, data: dict[str, str | None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


async def call_openrouter(
    client: httpx.AsyncClient,
    cfg: Config,
    api_key: str,
    title: str,
    selftext: str,
) -> str:
    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(title=title, selftext=selftext),
            },
        ],
        "temperature": 0.3,
    }
    url = f"{cfg.openrouter_base_url.rstrip('/')}/chat/completions"
    resp = await client.post(
        url,
        headers=_openrouter_headers(api_key),
        json=payload,
    )
    resp.raise_for_status()
    body = resp.json()
    choices = body.get("choices") or []
    if not choices:
        raise ValueError("OpenRouter response missing choices")
    content = (choices[0].get("message") or {}).get("content")
    if not content or not str(content).strip():
        raise ValueError("OpenRouter returned empty content")
    return str(content).strip()


async def summarize_one(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    cfg: Config,
    api_key: str,
    post_id: str,
    title: str,
    selftext: str,
) -> tuple[str, str | None]:
    async with sem:
        last_err: Exception | None = None
        for attempt in range(cfg.retry_attempts):
            try:
                text = await call_openrouter(client, cfg, api_key, title, selftext)
                return post_id, text
            except Exception as exc:
                last_err = exc
                log.warning(
                    "  post %s attempt %s/%s failed: %s",
                    post_id,
                    attempt + 1,
                    cfg.retry_attempts,
                    exc,
                )
                if attempt + 1 < cfg.retry_attempts:
                    await asyncio.sleep(cfg.retry_delay_s * (2**attempt))
        log.error("  post %s giving up after retries: %s", post_id, last_err)
        return post_id, None


async def run_pass1(cfg: Config, api_key: str) -> dict[str, str | None]:
    posts = load_indexed_posts(cfg)
    cache_path = cfg.data_dir / cfg.summaries_cache
    checkpoint: dict[str, str | None] = load_checkpoint(cache_path)

    pending: list[dict[str, Any]] = []
    for p in posts:
        pid = p["id"]
        if pid not in checkpoint:
            pending.append(p)
        elif checkpoint[pid] is None:
            # Retry failed entries
            pending.append(p)

    stats = Pass1Stats()
    stats.total_indexed_posts = len(posts)
    stats.already_cached = len(posts) - len(pending)

    log.info(
        "=== Pass 1: Summarize posts | indexed=%s | pending=%s | already_cached=%s ===",
        stats.total_indexed_posts,
        len(pending),
        stats.already_cached,
    )

    if not pending:
        log.info("Nothing to do — checkpoint complete.")
        return checkpoint

    sem = asyncio.Semaphore(cfg.max_concurrent)
    timeout = httpx.Timeout(cfg.request_timeout_s)
    limits = httpx.Limits(max_connections=cfg.max_concurrent + 5)

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        batch: list[dict[str, Any]] = []
        processed = 0

        async def flush_batch() -> None:
            nonlocal checkpoint, processed
            if not batch:
                return
            tasks = [
                summarize_one(
                    client,
                    sem,
                    cfg,
                    api_key,
                    b["id"],
                    b["title"],
                    b["selftext"],
                )
                for b in batch
            ]
            results = await asyncio.gather(*tasks)
            for post_id, summary in results:
                checkpoint[post_id] = summary
                stats.api_calls += 1
                if summary is not None:
                    stats.successes += 1
                else:
                    stats.failures += 1
            save_checkpoint(cache_path, checkpoint)
            processed += len(batch)
            batch.clear()
            if processed % cfg.progress_interval == 0 or processed == len(pending):
                log.info(
                    "  Pass 1 — %s/%s posts summarized (checkpoint saved)",
                    processed,
                    len(pending),
                )

        for p in pending:
            batch.append(p)
            if len(batch) >= cfg.checkpoint_batch_size:
                await flush_batch()
        await flush_batch()

    log.info(
        "Pass 1 done — api_calls=%s successes=%s failures=%s → %s",
        stats.api_calls,
        stats.successes,
        stats.failures,
        cache_path,
    )
    return checkpoint


def bare_post_id_from_link(link_id: str) -> str:
    if link_id.startswith("t3_"):
        return link_id[3:]
    return link_id


def run_pass2(cfg: Config, summaries: dict[str, str | None]) -> Pass2Stats:
    in_path = cfg.data_dir / cfg.comment_chunks_in
    out_path = cfg.data_dir / cfg.comment_chunks_out
    stats = Pass2Stats()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8") as fin, out_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            stats.total_chunks += 1

            link_id = rec.get("link_id") or ""
            bid = bare_post_id_from_link(link_id)
            summary = summaries.get(bid)
            if summary is not None:
                rec["post_summary"] = summary
                stats.with_summary += 1
            else:
                rec["post_summary"] = None
                stats.missing_lookup += 1

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    log.info(
        "Pass 2 done — chunks=%s with_summary=%s missing_lookup=%s → %s",
        stats.total_chunks,
        stats.with_summary,
        stats.missing_lookup,
        out_path,
    )
    return stats


def build_report(
    cfg: Config,
    summaries: dict[str, str | None],
    pass2: Pass2Stats,
) -> dict[str, Any]:
    non_null = sum(1 for v in summaries.values() if v is not None)
    null_count = sum(1 for v in summaries.values() if v is None)
    return {
        "schema": "enrich_report_v1",
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "parameters": {
            "model": cfg.model,
            "openrouter_base_url": cfg.openrouter_base_url,
        },
        "pass1": {
            "checkpoint_file": str(cfg.data_dir / cfg.summaries_cache),
            "post_ids_in_checkpoint": len(summaries),
            "summaries_non_null": non_null,
            "summaries_null": null_count,
        },
        "pass2": {
            "input": str(cfg.data_dir / cfg.comment_chunks_in),
            "output": str(cfg.data_dir / cfg.comment_chunks_out),
            "total_chunks": pass2.total_chunks,
            "chunks_with_post_summary": pass2.with_summary,
            "chunks_missing_lookup": pass2.missing_lookup,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich comment chunks with post summaries.")
    parser.add_argument(
        "--pass1-only",
        action="store_true",
        help="Only run LLM summarization (writes post_summaries.json)",
    )
    parser.add_argument(
        "--pass2-only",
        action="store_true",
        help="Only merge summaries into comment_chunks (reads checkpoint)",
    )
    args = parser.parse_args()
    if args.pass1_only and args.pass2_only:
        log.error("Cannot combine --pass1-only and --pass2-only")
        sys.exit(1)

    cfg = Config()
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key and not args.pass2_only:
        log.error("OPENROUTER_API_KEY is not set (check .env)")
        sys.exit(1)

    cfg.reports_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, str | None]
    if args.pass2_only:
        cache_path = cfg.data_dir / cfg.summaries_cache
        if not cache_path.exists():
            log.error("Checkpoint not found: %s", cache_path)
            sys.exit(1)
        summaries = load_checkpoint(cache_path)
    else:
        summaries = asyncio.run(run_pass1(cfg, api_key))

    if args.pass1_only:
        report = build_report(
            cfg,
            summaries,
            Pass2Stats(),
        )
        (cfg.reports_dir / cfg.report_out).write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log.info("Report → %s", cfg.reports_dir / cfg.report_out)
        return

    pass2 = run_pass2(cfg, summaries)

    report = build_report(cfg, summaries, pass2)
    report_path = cfg.reports_dir / cfg.report_out
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Report → %s", report_path)
    print(
        f"\n✅  Enrichment complete.\n"
        f"    Checkpoint: {cfg.data_dir / cfg.summaries_cache}\n"
        f"    Output:     {cfg.data_dir / cfg.comment_chunks_out}\n"
        f"    Verify, then: mv {cfg.comment_chunks_out} comment_chunks.jsonl\n"
    )


if __name__ == "__main__":
    main()
