#!/usr/bin/env python3
"""Fast safety validation for SEED RL training checkpoints.

This script is intended for ``--checkpoint-validation-command``.  It evaluates
the just-saved checkpoint against a small hard-opponent subset and returns a
non-zero exit code when the candidate shows an early worst-case collapse.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent


def _checkpoint_base(path: str) -> str:
    if path.endswith(".actor.weights.h5"):
        return path[: -len(".actor.weights.h5")]
    return path


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _opponent_name(row: dict) -> str:
    raw = row.get("opponent_name") or row.get("opponent") or "unknown"
    return os.path.basename(str(raw).replace(".actor.weights.h5", ""))


def _dedupe(paths: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for path in paths:
        base = _checkpoint_base(str(path))
        key = os.path.abspath(base)
        if key in seen:
            continue
        seen.add(key)
        out.append(base)
    return out


def _load_opponents(args: argparse.Namespace) -> list[str]:
    opponents: list[str] = []
    opponents.extend(args.opponent or [])

    scored_rows: list[dict] = []
    for path in args.opponent_file or []:
        scored_rows.extend(_read_jsonl(path))
    scored_rows.sort(key=lambda row: float(row.get("bb_per_100", 0.0)))

    for row in scored_rows[: max(0, args.worst_count)]:
        opponent = row.get("opponent")
        if opponent:
            opponents.append(str(opponent))

    if args.always_include:
        wanted = {
            os.path.basename(str(name).replace(".actor.weights.h5", ""))
            for name in args.always_include
        }
        for row in scored_rows:
            if _opponent_name(row) in wanted and row.get("opponent"):
                opponents.append(str(row["opponent"]))

    return _dedupe(opponents)


def _baseline_scores(paths: list[Path]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for path in paths:
        for row in _read_jsonl(path):
            scores[_opponent_name(row)] = float(row.get("bb_per_100", 0.0))
    return scores


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = _mean(values)
    return (sum((value - avg) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def _aggregate_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(_opponent_name(row), []).append(row)

    aggregated: list[dict] = []
    for name, group in grouped.items():
        base = dict(group[0])
        scores = [float(row.get("bb_per_100", 0.0)) for row in group]
        invalid_rates = [float(row.get("invalid_bet_rate", 0.0)) for row in group]
        trunc_rates = [float(row.get("truncated_hand_rate", 0.0)) for row in group]
        base["opponent_name"] = name
        base["bb_per_100"] = _mean(scores)
        base["bb_per_100_min"] = min(scores)
        base["bb_per_100_max"] = max(scores)
        base["bb_per_100_std"] = _std(scores)
        base["invalid_bet_rate"] = max(invalid_rates) if invalid_rates else 0.0
        base["invalid_bet_rate_avg"] = _mean(invalid_rates)
        base["truncated_hand_rate"] = max(trunc_rates) if trunc_rates else 0.0
        base["truncated_hand_rate_avg"] = _mean(trunc_rates)
        base["repeats"] = len(group)
        base["total_games"] = sum(int(row.get("total_games", 0)) for row in group)
        aggregated.append(base)
    return aggregated


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("SEED_RL_CHECKPOINT"),
        help="Candidate checkpoint base path. Defaults to SEED_RL_CHECKPOINT.",
    )
    parser.add_argument("--opponent", action="append", default=[])
    parser.add_argument(
        "--opponent-file",
        action="append",
        type=Path,
        default=[],
        help="JSONL matrix or aggregate file. Worst rows are used as hard opponents.",
    )
    parser.add_argument(
        "--always-include",
        action="append",
        default=[],
        help="Opponent basename that should be included when found in opponent-file.",
    )
    parser.add_argument("--worst-count", type=int, default=4)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Run multiple seeds and gate on per-opponent aggregate rows.",
    )
    parser.add_argument(
        "--seed-step",
        type=int,
        default=1,
        help="Seed increment between repeats.",
    )
    parser.add_argument("--hidden-dims", default="2048,2048,1024,512")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--expected-count", type=int, default=0)
    parser.add_argument("--min-avg", type=float, default=-150.0)
    parser.add_argument("--min-worst", type=float, default=-350.0)
    parser.add_argument("--fail-fast-worst", type=float, default=-600.0)
    parser.add_argument("--max-invalid", type=float, default=0.001)
    parser.add_argument("--max-truncated", type=float, default=0.005)
    parser.add_argument("--baseline-file", action="append", type=Path, default=[])
    parser.add_argument("--max-opponent-regression", type=float)
    parser.add_argument(
        "--reference-file",
        type=Path,
        help="Optional reference distillation .npz checked after opponent evaluation.",
    )
    parser.add_argument(
        "--max-reference-mean-kl",
        type=float,
        help="Fail if checkpoint mean KL against --reference-file exceeds this value.",
    )
    parser.add_argument(
        "--max-reference-p95-kl",
        type=float,
        help="Fail if checkpoint p95 KL against --reference-file exceeds this value.",
    )
    parser.add_argument("--pass-exit-code", type=int, default=0)
    parser.add_argument("--fail-exit-code", type=int, default=3)
    return parser.parse_args()


def _default_out_dir() -> Path:
    checkpoint_dir = os.environ.get("SEED_RL_CHECKPOINT_DIR")
    if checkpoint_dir:
        return Path(checkpoint_dir) / "safety_validation"
    return ROOT / "runs" / "safety_validation"


def _write_report(
    path: Path,
    checkpoint: str,
    raw_path: Path,
    rows: list[dict],
    failures: list[str],
) -> None:
    lines = [
        "# SEED RL Safety Validation",
        "",
        f"- checkpoint: `{checkpoint}`",
        f"- raw: `{raw_path}`",
        f"- status: `{'FAIL' if failures else 'PASS'}`",
        "",
        "## Rows",
        "",
        "| opponent | bb/100 | min | max | std | repeats | invalid | trunc |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: float(item.get("bb_per_100", 0.0))):
        lines.append(
            f"| {_opponent_name(row)} | "
            f"{float(row.get('bb_per_100', 0.0)):.1f} | "
            f"{float(row.get('bb_per_100_min', row.get('bb_per_100', 0.0))):.1f} | "
            f"{float(row.get('bb_per_100_max', row.get('bb_per_100', 0.0))):.1f} | "
            f"{float(row.get('bb_per_100_std', 0.0)):.1f} | "
            f"{int(row.get('repeats', 1))} | "
            f"{float(row.get('invalid_bet_rate', 0.0)):.2%} | "
            f"{float(row.get('truncated_hand_rate', 0.0)):.2%} |"
        )
    if failures:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {failure}" for failure in failures)
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not args.checkpoint:
        raise SystemExit("--checkpoint is required or SEED_RL_CHECKPOINT must be set")

    checkpoint = _checkpoint_base(args.checkpoint)
    opponents = [
        path for path in _load_opponents(args)
        if os.path.abspath(path) != os.path.abspath(checkpoint)
    ]
    if not opponents:
        raise SystemExit("No safety opponents found")

    out_dir = args.out_dir or _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    label = os.path.basename(checkpoint)
    total_hands = os.environ.get("SEED_RL_TOTAL_HANDS")
    stamp = total_hands or datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = out_dir / f"{label}_{stamp}.jsonl"
    report_path = out_dir / f"{label}_{stamp}_report.md"

    all_rows: list[dict] = []
    return_codes: list[int] = []
    repeats = max(1, int(args.repeats))
    for repeat_idx in range(repeats):
        repeat_raw_path = (
            raw_path
            if repeats == 1
            else out_dir / f"{label}_{stamp}_repeat{repeat_idx + 1}.jsonl"
        )
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "evaluate_seed_pool.py"),
            "--model",
            checkpoint,
            "--games",
            str(args.games),
            "--seed",
            str(args.seed + repeat_idx * args.seed_step),
            "--hidden-dims",
            args.hidden_dims,
            "--out",
            str(repeat_raw_path),
            "--fail-fast-worst-bb100",
            str(args.fail_fast_worst),
            "--max-invalid-rate",
            str(args.max_invalid),
            "--max-truncated-rate",
            str(args.max_truncated),
            "--continue-on-gate-fail",
            "--fail-exit-code",
            str(args.fail_exit_code),
        ]
        for opponent in opponents:
            cmd.extend(["--opponent", opponent])

        print("+ " + " ".join(cmd), flush=True)
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        return_codes.append(rc)
        all_rows.extend(_read_jsonl(repeat_raw_path))

    rows = _aggregate_rows(all_rows) if repeats > 1 else all_rows
    if repeats > 1:
        _write_jsonl(raw_path, rows)
    failures: list[str] = []
    if args.expected_count and len(rows) < args.expected_count:
        failures.append(f"only {len(rows)}/{args.expected_count} opponents evaluated")
    if rows:
        scores = [float(row.get("bb_per_100", 0.0)) for row in rows]
        avg = sum(scores) / len(scores)
        worst = min(rows, key=lambda row: float(row.get("bb_per_100", 0.0)))
        if avg < args.min_avg:
            failures.append(f"avg {avg:.1f} < min {args.min_avg:.1f}")
        if float(worst.get("bb_per_100", 0.0)) < args.min_worst:
            failures.append(
                f"worst {_opponent_name(worst)} {float(worst.get('bb_per_100', 0.0)):.1f} "
                f"< min {args.min_worst:.1f}"
            )
        for row in rows:
            invalid = float(row.get("invalid_bet_rate", 0.0))
            truncated = float(row.get("truncated_hand_rate", 0.0))
            if invalid > args.max_invalid:
                failures.append(f"{_opponent_name(row)} invalid {invalid:.4f} > {args.max_invalid:.4f}")
            if truncated > args.max_truncated:
                failures.append(
                    f"{_opponent_name(row)} truncated {truncated:.4f} > {args.max_truncated:.4f}"
                )

        if args.baseline_file and args.max_opponent_regression is not None:
            baseline = _baseline_scores(args.baseline_file)
            for row in rows:
                name = _opponent_name(row)
                if name not in baseline:
                    continue
                delta = float(row.get("bb_per_100", 0.0)) - baseline[name]
                if delta < -args.max_opponent_regression:
                    failures.append(
                        f"{name} regression {delta:.1f} < -{args.max_opponent_regression:.1f}"
                    )
    else:
        failures.append("no result rows written")

    bad_rcs = [rc for rc in return_codes if rc not in (0, args.fail_exit_code)]
    if bad_rcs:
        failures.append(f"evaluate_seed_pool exited {bad_rcs}")

    if args.reference_file:
        reference_out = out_dir / f"{label}_{stamp}_reference_kl.json"
        ref_cmd = [
            sys.executable,
            str(SCRIPT_DIR / "check_seed_reference_kl.py"),
            "--model",
            checkpoint,
            "--reference-file",
            str(args.reference_file),
            "--hidden-dims",
            args.hidden_dims,
            "--out",
            str(reference_out),
            "--fail-exit-code",
            str(args.fail_exit_code),
        ]
        if args.max_reference_mean_kl is not None:
            ref_cmd.extend(["--max-mean-kl", str(args.max_reference_mean_kl)])
        if args.max_reference_p95_kl is not None:
            ref_cmd.extend(["--max-p95-kl", str(args.max_reference_p95_kl)])
        print("+ " + " ".join(ref_cmd), flush=True)
        ref_rc = subprocess.run(ref_cmd, cwd=ROOT).returncode
        try:
            ref_row = json.loads(reference_out.read_text(encoding="utf-8"))
            if ref_row.get("failures"):
                failures.extend(f"reference_kl: {item}" for item in ref_row["failures"])
        except Exception as exc:
            failures.append(f"reference_kl report unreadable: {exc}")
        if ref_rc not in (0, args.fail_exit_code):
            failures.append(f"check_seed_reference_kl exited {ref_rc}")

    _write_report(report_path, checkpoint, raw_path, rows, failures)
    print(f"Safety report: {report_path}", flush=True)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", flush=True)
        return args.fail_exit_code
    print("PASS: safety validation passed", flush=True)
    return args.pass_exit_code


if __name__ == "__main__":
    raise SystemExit(main())
