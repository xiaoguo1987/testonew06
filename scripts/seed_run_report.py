#!/usr/bin/env python3
"""Summarize SEED RL metrics.jsonl into a compact Markdown report."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def num(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def latest_with(rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    for row in reversed(rows):
        if key in row:
            return row
    return None


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round((len(values) - 1) * q)))
    return values[idx]


def action_line(dist: dict[str, float]) -> str:
    if not dist:
        return "n/a"
    keys = [
        "fold", "call", "raise_25pct", "raise_33pct", "raise_50pct",
        "raise_66pct", "raise_75pct", "raise_100pct", "raise_125pct",
        "raise_150pct", "raise_200pct", "allin",
    ]
    return ", ".join(f"{key}={pct(dist.get(key))}" for key in keys if key in dist)


def build_report(metrics_path: Path, title: str) -> str:
    rows = load_rows(metrics_path)
    ppos = [row for row in rows if row.get("event") == "ppo_update"]
    evals = [row for row in rows if row.get("event") in {"evaluation", "final_evaluation"}]
    complete = latest_with(rows, "elapsed_seconds")
    dist_row = latest_with(ppos, "train_action_distribution")

    kls = [float(row["last_kl"]) for row in ppos if isinstance(row.get("last_kl"), (int, float))]
    interval_hps = [
        float(row["interval_hps"])
        for row in ppos
        if isinstance(row.get("interval_hps"), (int, float))
    ]

    lines = [
        f"# {title}",
        "",
        f"- Metrics: `{metrics_path}`",
        f"- Events: total={len(rows)}, ppo_update={len(ppos)}, evaluation={len(evals)}",
    ]

    if complete:
        lines.extend([
            f"- Complete hands: {num(complete.get('total_hands'), 0)}",
            f"- Elapsed seconds: {num(complete.get('elapsed_seconds'), 1)}",
            f"- Overall hand/s: {num(complete.get('hps'), 1)}",
            f"- PPO updates: {num(complete.get('ppo_updates'), 0)}",
            f"- Best score: {num(complete.get('best_score'), 3)}",
        ])

    if kls:
        lines.append(
            f"- KL: median={num(statistics.median(kls), 4)}, "
            f"p95={num(quantile(kls, 0.95), 4)}, max={num(max(kls), 4)}"
        )
    if interval_hps:
        lines.append(
            f"- Interval hand/s: median={num(statistics.median(interval_hps), 1)}, "
            f"p95={num(quantile(interval_hps, 0.95), 1)}"
        )

    if dist_row:
        lines.extend([
            "",
            "## Action Distribution",
            "",
            f"- Cumulative: {action_line(dist_row.get('train_action_distribution', {}))}",
            f"- Last interval: {action_line(dist_row.get('train_action_distribution_interval', {}))}",
        ])

    if evals:
        lines.extend(["", "## Evaluations", ""])
        for row in evals[-10:]:
            lines.append(
                "- "
                f"hands={num(row.get('total_hands'), 0)}, "
                f"policy={row.get('primary_opponent_policy', 'n/a')}, "
                f"win={pct(row.get('win_rate'))}, "
                f"bb/100={num(row.get('bb_per_100'), 1)}, "
                f"invalid={pct(row.get('invalid_bet_rate'))}, "
                f"truncated={pct(row.get('truncated_hand_rate'))}"
            )

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--title", default="SEED RL Run Report")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = build_report(args.metrics, args.title)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
