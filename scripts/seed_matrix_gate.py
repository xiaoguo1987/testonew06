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
    avg_lcb95: float
    worst_lcb95: float
    worst_lcb95_opponent: str
    max_score_std: float
    max_score_std_opponent: str
    invalid_rate: float
    truncated_rate: float
    max_invalid_rate: float
    max_invalid_opponent: str
    max_truncated_rate: float
    max_truncated_opponent: str
    opponent_scores: dict[str, float]


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


def _opponent_key(row: dict[str, Any]) -> str:
    raw = row.get("opponent_name") or row.get("opponent") or "unknown"
    return os.path.basename(str(raw))


def _row_lcb95(row: dict[str, Any]) -> float:
    if isinstance(row.get("bb_per_100_lcb95"), (int, float)):
        return float(row["bb_per_100_lcb95"])
    score = float(row.get("bb_per_100", 0.0))
    std = float(row.get("bb_per_100_std", 0.0))
    repeats = max(int(row.get("repeats", 1)), 1)
    return score - 1.96 * (std / math.sqrt(repeats))


def summarize(path: Path) -> MatrixSummary:
    rows = _read_rows(path)
    if not rows:
        raise ValueError(f"{path}: no matrix rows found")

    for row in rows:
        if not isinstance(row.get("bb_per_100"), (int, float)):
            raise ValueError(f"{path}: row missing numeric bb_per_100")

    worst = min(rows, key=lambda row: float(row["bb_per_100"]))
    best = max(rows, key=lambda row: float(row["bb_per_100"]))
    worst_lcb = min(rows, key=_row_lcb95)
    max_std = max(rows, key=lambda row: float(row.get("bb_per_100_std", 0.0)))
    avg = sum(float(row["bb_per_100"]) for row in rows) / len(rows)
    avg_lcb95 = sum(_row_lcb95(row) for row in rows) / len(rows)
    invalid = sum(float(row.get("invalid_bet_rate", 0.0)) for row in rows) / len(rows)
    truncated = sum(float(row.get("truncated_hand_rate", 0.0)) for row in rows) / len(rows)
    max_invalid = max(rows, key=lambda row: float(row.get("invalid_bet_rate", 0.0)))
    max_truncated = max(rows, key=lambda row: float(row.get("truncated_hand_rate", 0.0)))
    opponent_scores = {
        _opponent_key(row): float(row["bb_per_100"])
        for row in rows
    }
    return MatrixSummary(
        label=_label_from_rows(path, rows),
        path=path,
        count=len(rows),
        avg_bb100=avg,
        worst_bb100=float(worst["bb_per_100"]),
        worst_opponent=str(worst.get("opponent_name") or os.path.basename(str(worst.get("opponent", "")))),
        best_bb100=float(best["bb_per_100"]),
        best_opponent=str(best.get("opponent_name") or os.path.basename(str(best.get("opponent", "")))),
        avg_lcb95=avg_lcb95,
        worst_lcb95=_row_lcb95(worst_lcb),
        worst_lcb95_opponent=_opponent_key(worst_lcb),
        max_score_std=float(max_std.get("bb_per_100_std", 0.0)),
        max_score_std_opponent=_opponent_key(max_std),
        invalid_rate=invalid,
        truncated_rate=truncated,
        max_invalid_rate=float(max_invalid.get("invalid_bet_rate", 0.0)),
        max_invalid_opponent=_opponent_key(max_invalid),
        max_truncated_rate=float(max_truncated.get("truncated_hand_rate", 0.0)),
        max_truncated_opponent=_opponent_key(max_truncated),
        opponent_scores=opponent_scores,
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
    if args.min_avg_lcb is not None and summary.avg_lcb95 < args.min_avg_lcb:
        failures.append(
            f"avg lcb95 {_fmt(summary.avg_lcb95)} < min {_fmt(args.min_avg_lcb)}"
        )
    if summary.worst_bb100 < args.min_worst:
        failures.append(f"worst {_fmt(summary.worst_bb100)} < min {_fmt(args.min_worst)}")
    if args.max_row_std is not None and summary.max_score_std > args.max_row_std:
        failures.append(
            f"row std {summary.max_score_std_opponent} {_fmt(summary.max_score_std)} "
            f"> max {_fmt(args.max_row_std)}"
        )
    if summary.invalid_rate > args.max_invalid:
        failures.append(f"invalid {_pct(summary.invalid_rate)} > max {_pct(args.max_invalid)}")
    if summary.truncated_rate > args.max_truncated:
        failures.append(f"truncated {_pct(summary.truncated_rate)} > max {_pct(args.max_truncated)}")
    if args.max_row_invalid is not None and summary.max_invalid_rate > args.max_row_invalid:
        failures.append(
            f"row invalid {summary.max_invalid_opponent} {_pct(summary.max_invalid_rate)} "
            f"> max {_pct(args.max_row_invalid)}"
        )
    if args.max_row_truncated is not None and summary.max_truncated_rate > args.max_row_truncated:
        failures.append(
            f"row truncated {summary.max_truncated_opponent} {_pct(summary.max_truncated_rate)} "
            f"> max {_pct(args.max_row_truncated)}"
        )
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
    if baseline and args.max_opponent_regression is not None:
        regressions: list[tuple[str, float]] = []
        for opponent, baseline_score in baseline.opponent_scores.items():
            if opponent not in summary.opponent_scores:
                continue
            delta = summary.opponent_scores[opponent] - baseline_score
            if delta < -args.max_opponent_regression:
                regressions.append((opponent, delta))
        if regressions:
            regressions.sort(key=lambda item: item[1])
            shown = ", ".join(
                f"{opponent} delta {_fmt(delta)}"
                for opponent, delta in regressions[:3]
            )
            if len(regressions) > 3:
                shown += f", and {len(regressions) - 3} more"
            failures.append(
                f"per-opponent regression > {_fmt(args.max_opponent_regression)}: {shown}"
            )
    return failures


def build_report(
    candidates: list[MatrixSummary],
    args: argparse.Namespace,
    baseline: MatrixSummary | None,
) -> tuple[str, bool]:
    ranked = sorted(candidates, key=lambda item: (item.avg_bb100, item.worst_bb100), reverse=True)
    regression_threshold = (
        _fmt(args.max_opponent_regression)
        if args.max_opponent_regression is not None
        else "disabled"
    )
    max_row_invalid = _pct(args.max_row_invalid) if args.max_row_invalid is not None else "disabled"
    max_row_truncated = _pct(args.max_row_truncated) if args.max_row_truncated is not None else "disabled"
    min_avg_lcb = _fmt(args.min_avg_lcb) if args.min_avg_lcb is not None else "disabled"
    max_row_std = _fmt(args.max_row_std) if args.max_row_std is not None else "disabled"
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
                f"avg_lcb95={_fmt(baseline.avg_lcb95)}, "
                f"worst={baseline.worst_opponent} {_fmt(baseline.worst_bb100)}, "
                f"max_std={baseline.max_score_std_opponent} {_fmt(baseline.max_score_std)}, "
                f"invalid={_pct(baseline.invalid_rate)}, trunc={_pct(baseline.truncated_rate)}"
            ),
            "",
        ])

    lines.extend([
        "## Thresholds",
        "",
        f"- min_avg={_fmt(args.min_avg)}",
        f"- min_avg_lcb={min_avg_lcb}",
        f"- min_worst={_fmt(args.min_worst)}",
        f"- max_row_std={max_row_std}",
        f"- fail_fast_worst={_fmt(args.fail_fast_worst)}",
        f"- max_invalid={_pct(args.max_invalid)}",
        f"- max_truncated={_pct(args.max_truncated)}",
        f"- max_row_invalid={max_row_invalid}",
        f"- max_row_truncated={max_row_truncated}",
        f"- max_opponent_regression={regression_threshold}",
        f"- expected_count={args.expected_count or 'disabled'}",
        "",
        "## Candidates",
        "",
        "| status | model | count | avg | avg lcb95 | worst | max std | best | invalid | trunc | max invalid | max trunc | notes |",
        "|---|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---|",
    ])

    for summary in ranked:
        failures = _check_summary(summary, args, baseline)
        status = "PASS" if not failures else "FAIL"
        any_pass = any_pass or not failures
        notes = "; ".join(failures) if failures else "accepted by configured gates"
        lines.append(
            "| "
            f"{status} | {summary.label} | {summary.count} | {_fmt(summary.avg_bb100)} | "
            f"{_fmt(summary.avg_lcb95)} | "
            f"{summary.worst_opponent} {_fmt(summary.worst_bb100)} | "
            f"{summary.max_score_std_opponent} {_fmt(summary.max_score_std)} | "
            f"{summary.best_opponent} {_fmt(summary.best_bb100)} | "
            f"{_pct(summary.invalid_rate)} | {_pct(summary.truncated_rate)} | "
            f"{summary.max_invalid_opponent} {_pct(summary.max_invalid_rate)} | "
            f"{summary.max_truncated_opponent} {_pct(summary.max_truncated_rate)} | "
            f"{notes} |"
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
                f"avg_lcb95={_fmt(best_passing.avg_lcb95)}, "
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
                f"avg_lcb95={_fmt(best.avg_lcb95)}, "
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
    parser.add_argument(
        "--min-avg-lcb",
        type=float,
        default=None,
        help="Fail if the average per-opponent 95% lower confidence bound is below this bb/100",
    )
    parser.add_argument("--min-worst", type=float, default=-360.1)
    parser.add_argument(
        "--max-row-std",
        type=float,
        default=None,
        help="Fail if any repeated-evaluation opponent row exceeds this bb/100 stddev",
    )
    parser.add_argument("--fail-fast-worst", type=float, default=-500.0)
    parser.add_argument("--max-invalid", type=float, default=0.001)
    parser.add_argument("--max-truncated", type=float, default=0.005)
    parser.add_argument(
        "--max-row-invalid",
        type=float,
        default=None,
        help="Fail if any single opponent row exceeds this invalid bet rate",
    )
    parser.add_argument(
        "--max-row-truncated",
        type=float,
        default=None,
        help="Fail if any single opponent row exceeds this truncated hand rate",
    )
    parser.add_argument(
        "--max-opponent-regression",
        type=float,
        default=None,
        help="With --baseline, fail if any matched opponent drops by more than this bb/100",
    )
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
