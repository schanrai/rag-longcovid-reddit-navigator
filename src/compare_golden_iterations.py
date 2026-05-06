#!/usr/bin/env python3
"""
Compare aggregate scores across golden_iteration_*.json reports.

Typical use after swapping --second-judge-model: compare secondary aggregates between runs.

Usage:
    cd projects/rag-longcovid-reddit-navigator
    python -m src.compare_golden_iterations \\
        reports/synthesis_eval/golden_iteration_1.json \\
        reports/synthesis_eval/golden_iteration_2.json

    python -m src.compare_golden_iterations --latest 2
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

AGG_KEYS = [
    "instruction_adherence",
    "citation_accuracy",
    "format_consistency",
    "tone_intent",
    "diversity",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = PROJECT_ROOT / "reports" / "synthesis_eval"


def _iteration_num(path: Path) -> int:
    m = re.match(r"golden_iteration_(\d+)\.json$", path.name)
    return int(m.group(1)) if m else -1


def _collect_reports(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    paths = sorted(directory.glob("golden_iteration_*.json"), key=_iteration_num)
    return paths


def _row(label: str, agg: dict[str, float | int], *, width: int = 22) -> str:
    parts = [f"{label:<{width}}"]
    for k in AGG_KEYS:
        parts.append(f"{agg.get(k, 0):>7.1f}")
    parts.append(f"{agg.get('mean', 0):>7.1f}")
    return "  ".join(parts)


def _print_header(width: int) -> None:
    hdr = [f"{'report':<{width}}"]
    for k in AGG_KEYS:
        hdr.append(f"{k[:7]:>7}")
    hdr.append(f"{'mean':>7}")
    print("  ".join(hdr))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare aggregates between two golden_iteration JSON reports.",
    )
    ap.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Exactly two report paths (older first recommended). Ignored if --latest is set.",
    )
    ap.add_argument(
        "--latest",
        type=int,
        metavar="N",
        help=f"Use the N newest files from --dir (default: {DEFAULT_DIR}).",
    )
    ap.add_argument(
        "--dir",
        type=Path,
        default=DEFAULT_DIR,
        help=f"Directory for --latest (default: {DEFAULT_DIR})",
    )
    args = ap.parse_args()

    if args.latest is not None:
        if args.latest < 2:
            raise SystemExit("--latest must be >= 2")
        all_p = _collect_reports(args.dir.resolve())
        if len(all_p) < args.latest:
            raise SystemExit(
                f"Need at least {args.latest} golden_iteration_*.json in {args.dir}; "
                f"found {len(all_p)}."
            )
        paths = all_p[-args.latest :]
    else:
        if len(args.paths) != 2:
            raise SystemExit("Provide exactly two report paths, or use --latest N")
        paths = [p.resolve() for p in args.paths]

    reports = []
    for p in paths:
        if not p.is_file():
            raise SystemExit(f"Not found: {p}")
        reports.append(json.loads(p.read_text(encoding="utf-8")))

    r_a, r_b = reports[0], reports[1]
    width = 36

    print()
    print(f"A: {paths[0].name}  (iteration {r_a.get('iteration')})")
    print(f"   synthesis={r_a.get('synthesis_model')}  judge={r_a.get('judge_model')}")
    if r_a.get("judge_model_secondary"):
        print(f"   secondary_judge={r_a.get('judge_model_secondary')}")
    print(f"B: {paths[1].name}  (iteration {r_b.get('iteration')})")
    print(f"   synthesis={r_b.get('synthesis_model')}  judge={r_b.get('judge_model')}")
    if r_b.get("judge_model_secondary"):
        print(f"   secondary_judge={r_b.get('judge_model_secondary')}")

    print("\n--- Primary judge aggregate (same rubric; scores differ if answers differ) ---")
    _print_header(width)
    agg_a = r_a.get("aggregate") or {}
    agg_b = r_b.get("aggregate") or {}
    print(_row("A primary", agg_a, width=width))
    print(_row("B primary", agg_b, width=width))
    if all(isinstance(agg_a.get(k), (int, float)) for k in AGG_KEYS):
        delta_p = {k: round(float(agg_b[k]) - float(agg_a[k]), 2) for k in AGG_KEYS}
        delta_p["mean"] = round(float(agg_b.get("mean", 0)) - float(agg_a.get("mean", 0)), 2)
        print(_row("Δ B−A", delta_p, width=width))

    sec_a = r_a.get("aggregate_secondary")
    sec_b = r_b.get("aggregate_secondary")
    print("\n--- Secondary judge aggregate (judge swap calibration) ---")
    if not sec_a and not sec_b:
        print("(no aggregate_secondary in either file — run with --second-judge-model)")
        return
    _print_header(width)
    if sec_a:
        print(_row("A secondary", sec_a, width=width))
    else:
        print(f"{'A secondary':<{width}}  (missing)")
    if sec_b:
        print(_row("B secondary", sec_b, width=width))
    else:
        print(f"{'B secondary':<{width}}  (missing)")
    if sec_a and sec_b:
        delta_s = {
            k: round(float(sec_b[k]) - float(sec_a[k]), 2) for k in AGG_KEYS
        }
        delta_s["mean"] = round(
            float(sec_b.get("mean", 0)) - float(sec_a.get("mean", 0)), 2
        )
        print(_row("Δ B−A", delta_s, width=width))
        mean_shift = delta_s["mean"]
        print(
            f"\nSecondary mean shift (B − A): {mean_shift:+.2f} — "
            f"{'material' if abs(mean_shift) >= 0.3 else 'small'} "
            f"(≥0.3 suggested threshold for ‘noticeable’)."
        )

    print()


if __name__ == "__main__":
    main()
