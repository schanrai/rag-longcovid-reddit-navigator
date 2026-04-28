"""
Clone a Weaviate collection to a new name, copying all objects and vectors.

Usage:
    python -m src.clone_collection \
        --src LongCovidChunks \
        --dst LongCovidChunks_depth_aware_v1 \
        [--save-jsonl data/exports/LongCovidChunks_depth_aware_v1.jsonl]

The --save-jsonl flag is optional but recommended: it writes a local JSONL backup
of all objects + vectors so you have a disk copy independent of Weaviate.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import weaviate
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def clone_collection(
    src_name: str,
    dst_name: str,
    save_jsonl: Path | None = None,
) -> None:
    from weaviate.classes.init import AdditionalConfig, Timeout

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=__import__("os").environ["WEAVIATE_URL"],
        auth_credentials=weaviate.auth.AuthApiKey(
            __import__("os").environ["WEAVIATE_API_KEY"]
        ),
        additional_config=AdditionalConfig(timeout=Timeout(init=60, query=120, insert=300)),
    )

    try:
        # ── Sanity checks ───────────────────────────────────────────────────
        if not client.collections.exists(src_name):
            raise ValueError(f"Source collection '{src_name}' does not exist")
        if client.collections.exists(dst_name):
            raise ValueError(
                f"Destination collection '{dst_name}' already exists — "
                "delete it first or choose a different name"
            )

        # ── Export schema and create destination ────────────────────────────
        log.info("Exporting schema from '%s'", src_name)
        config = client.collections.export_config(src_name)
        config.name = dst_name  # mutate name in place
        log.info("Creating destination collection '%s'", dst_name)
        client.collections.create_from_config(config)

        src = client.collections.get(src_name)
        dst = client.collections.get(dst_name)

        # ── Count source objects ─────────────────────────────────────────────
        total = src.aggregate.over_all(total_count=True).total_count
        log.info("Source has %d objects — starting copy", total)

        # ── Optional local JSONL backup ──────────────────────────────────────
        jsonl_fh = None
        if save_jsonl:
            save_jsonl.parent.mkdir(parents=True, exist_ok=True)
            jsonl_fh = open(save_jsonl, "w", encoding="utf-8")
            log.info("Local JSONL backup → %s", save_jsonl)

        # ── Copy objects in batches ──────────────────────────────────────────
        copied = 0
        failed = 0
        t_start = time.time()

        with dst.batch.dynamic() as batch:
            for obj in src.iterator(include_vector=True):
                vector = obj.vector.get("default") if isinstance(obj.vector, dict) else obj.vector

                batch.add_object(
                    properties=obj.properties,
                    vector=vector,
                    uuid=obj.uuid,  # preserve original UUIDs
                )

                if jsonl_fh:
                    jsonl_fh.write(
                        json.dumps(
                            {
                                "uuid": str(obj.uuid),
                                "properties": obj.properties,
                                "vector": vector,
                            },
                            default=str,
                        )
                        + "\n"
                    )

                copied += 1
                if copied % 10_000 == 0:
                    elapsed = time.time() - t_start
                    rate = copied / elapsed
                    remaining = (total - copied) / rate if rate > 0 else 0
                    log.info(
                        "  %d / %d copied  (%.0f obj/s  ~%.0f min remaining)",
                        copied, total, rate, remaining / 60,
                    )

        if jsonl_fh:
            jsonl_fh.close()

        # ── Verify ───────────────────────────────────────────────────────────
        dst_count = dst.aggregate.over_all(total_count=True).total_count
        log.info(
            "Clone complete — src=%d  dst=%d  failed=%d  elapsed=%.1fs",
            total, dst_count, failed, time.time() - t_start,
        )
        if dst_count != total:
            log.warning(
                "Count mismatch: src=%d dst=%d — inspect batch errors",
                total, dst_count,
            )
        else:
            log.info("✅ '%s' → '%s' verified", src_name, dst_name)

    finally:
        client.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Clone a Weaviate collection to a new name")
    ap.add_argument("--src", required=True, help="Source collection name")
    ap.add_argument("--dst", required=True, help="Destination collection name")
    ap.add_argument(
        "--save-jsonl",
        type=Path,
        default=None,
        help="Optional: also write a local JSONL backup of all objects + vectors",
    )
    args = ap.parse_args()

    clone_collection(
        src_name=args.src,
        dst_name=args.dst,
        save_jsonl=args.save_jsonl,
    )


if __name__ == "__main__":
    main()
