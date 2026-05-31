#!/usr/bin/env python3
"""Gate SEED RL matrix-evaluation JSONL files before accepting checkpoints.

The matrix evaluator writes one JSON object per opponent.  This script reduces
those rows to candidate-level summaries and applies retention gates so a model
cannot be accepted just because its average is good while one opponent regresses
badly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MatrixSummary:
    label: str
    path: Path
    count: int
    avg_bb100: float
    worst_bb100: float
    worst_opponent: str
    best_bb100: float
    best_opponent: str
    invalid_rate: float
    truncated_rate: float


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _label_from_rows(path: Path, rows: list[dict[str, Any]]) -> str:
    if rows and rows[0].get("model"):
        return os.path.basename(str(rows[0]["model"]))
    return path.stem


def summarize(path: Path) -> MatrixSummary:
    rows = _read_rows(path)
    if not rows:
        raise ValueError(f"{path}: no matrix rows found")

    for row in rows:
        if not isinstance(row.get("bb_per_100"), (int, float)):
            raise ValueError(f"{path}: row missing numeric bb_per_100")

    worst = min(rows, key=lambda row: float(row["bb_per_100"]))
    best = max(rows, key=lambda row: float(row["bb_per_100"]))
    avg = sum(float(row["bb_per_100"]) for row in rows) / len(rows)
    invalid = sum(float(row.get("invalid_bet_rate", 0.0)) for row in rows) / len(rows)
    truncated = sum(float(row.get("truncated_hand_rate", 0.0)) for row in rows) / len(rows)
    return MatrixSummary(
        label=_label_from_rows(path, rows),
        path=path,
        count=len(rows),
        avg_bb100=avg,
        worst_bb100=float(worst["bb_per_100"]),
        worst_opponent=str(worst.get("opponent_name") or os.path.basename(str(worst.get("opponent", "")))),
        best_bb100=float(best["bb_per_100"]),
        best_opponent=str(best.get("opponent_name") or os.path.basename(str(best.get("opponent", "")))),
        invalid_rate=invalid,
        truncated_rate=truncated,
    )


def _fmt(value: float, digits: int = 1) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _check_summary(
    summary: MatrixSummary,
    args: argparse.Namespace,
    baseline: MatrixSummary | None,
) -> list[str]:
    failures: list[str] = []
    if args.expected_count and summary.count < args.expected_count:
        failures.append(f"only {summary.count}/{args.expected_count} opponents evaluated")
    if summary.avg_bb100 < args.min_avg:
        failures.append(f"avg {_fmt(summary.avg_bb100)} < min {_fmt(args.min_avg)}")
    if summary.worst_bb100 < args.min_worst:
        failures.append(f"worst {_fmt(summary.worst_bb100)} < min {_fmt(args.min_worst)}")
    if summary.invalid_rate > args.max_invalid:
        failures.append(f"invalid {_pct(summary.invalid_rate)} > max {_pct(args.max_invalid)}")
    if summary.truncated_rate > args.max_truncated:
        failures.append(f"truncated {_pct(summary.truncated_rate)} > max {_pct(args.max_truncated)}")
    if summary.worst_bb100 < args.fail_fast_worst:
        failures.append(f"fail-fast worst {_fmt(summary.worst_bb100)} < {_fmt(args.fail_fast_worst)}")

    if baseline and args.require_baseline_improvement:
        avg_delta = summary.avg_bb100 - baseline.avg_bb100
        worst_delta = summary.worst_bb100 - baseline.worst_bb100
        if avg_delta < args.min_avg_delta:
            failures.append(
                f"avg delta {_fmt(avg_delta)} < required {_fmt(args.min_avg_delta)}"
            )
        if worst_delta < args.min_worst_delta:
            failures.append(
                f"worst delta {_fmt(worst_delta)} < required {_fmt(args.min_worst_delta)}"
            )
    return failures


def build_report(
    candidates: list[MatrixSummary],
    args: argparse.Namespace,
    baseline: MatrixSummary | None,
) -> tuple[str, bool]:
    ranked = sorted(candidates, key=lambda item: (item.avg_bb100, item.worst_bb100), reverse=True)
    any_pass = False
    lines = [
        "# SEED RL Matrix Gate",
        "",
    ]

    if baseline:
        lines.extend([
            "## Baseline",
            "",
            (
                f"- {baseline.label}: avg={_fmt(baseline.avg_bb100)}, "
                f"worst={baseline.worst_opponent} {_fmt(baseline.worst_bb100)}, "
                f"invalid={_pct(baseline.invalid_rate)}, trunc={_pct(baseline.truncated_rate)}"
            ),
            "",
        ])

    lines.extend([
        "## Thresholds",
        "",
        f"- min_avg={_fmt(args.min_avg)}",
        f"- min_worst={_fmt(args.min_worst)}",
        f"- fail_fast_worst={_fmt(args.fail_fast_worst)}",
        f"- max_invalid={_pct(args.max_invalid)}",
        f"- max_truncated={_pct(args.max_truncated)}",
        f"- expected_count={args.expected_count or 'disabled'}",
        "",
        "## Candidates",
        "",
        "| status | model | count | avg | worst | best | invalid | trunc | notes |",
        "|---|---:|---:|---:|---|---|---:|---:|---|",
    ])

    for summary in ranked:
        failures = _check_summary(summary, args, baseline)
        status = "PASS" if not failures else "FAIL"
        any_pass = any_pass or not failures
        notes = "; ".join(failures) if failures else "accepted by configured gates"
        lines.append(
            "| "
            f"{status} | {summary.label} | {summary.count} | {_fmt(summary.avg_bb100)} | "
            f"{summary.worst_opponent} {_fmt(summary.worst_bb100)} | "
            f"{summary.best_opponent} {_fmt(summary.best_bb100)} | "
            f"{_pct(summary.invalid_rate)} | {_pct(summary.truncated_rate)} | {notes} |"
        )

    passing = [
        summary for summary in ranked
        if not _check_summary(summary, args, baseline)
    ]
    if passing:
        best_passing = passing[0]
        lines.extend([
            "",
            "## Best Passing Candidate",
            "",
            (
                f"- {best_passing.label}: avg={_fmt(best_passing.avg_bb100)}, "
                f"worst={best_passing.worst_opponent} {_fmt(best_passing.worst_bb100)}"
            ),
        ])
    else:
        lines.extend([
            "",
            "## Best Passing Candidate",
            "",
            "- None. Keep the current baseline or run a different repair strategy.",
        ])

    if ranked:
        best = ranked[0]
        lines.extend([
            "",
            "## Best By Average",
            "",
            (
                f"- {best.label}: avg={_fmt(best.avg_bb100)}, "
                f"worst={best.worst_opponent} {_fmt(best.worst_bb100)}"
            ),
        ])

    lines.append("")
    return "\n".join(lines), any_pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", nargs="+", type=Path, help="Candidate matrix JSONL file(s)")
    parser.add_argument("--baseline", type=Path, help="Optional baseline matrix JSONL")
    parser.add_argument("--out", type=Path, help="Optional Markdown report path")
    parser.add_argument("--expected-count", type=int, default=13)
    parser.add_argument("--min-avg", type=float, default=103.9)
    parser.add_argument("--min-worst", type=float, default=-360.1)
    parser.add_argument("--fail-fast-worst", type=float, default=-500.0)
    parser.add_argument("--max-invalid", type=float, default=0.001)
    parser.add_argument("--max-truncated", type=float, default=0.005)
    parser.add_argument("--require-baseline-improvement", action="store_true")
    parser.add_argument("--min-avg-delta", type=float, default=0.0)
    parser.add_argument("--min-worst-delta", type=float, default=0.0)
    parser.add_argument(
        "--pass-exit-code",
        type=int,
        default=0,
        help="Exit code when at least one candidate passes",
    )
    parser.add_argument(
        "--fail-exit-code",
        type=int,
        default=1,
        help="Exit code when no candidate passes",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        baseline = summarize(args.baseline) if args.baseline else None
        candidates = [summarize(path) for path in args.matrix]
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"seed_matrix_gate: {exc}", file=sys.stderr)
        return 2
    report, any_pass = build_report(candidates, args, baseline)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
    else:
        print(report)
    return args.pass_exit_code if any_pass else args.fail_exit_code


if __name__ == "__main__":
    raise SystemExit(main())
