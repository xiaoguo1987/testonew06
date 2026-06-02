#!/usr/bin/env python3
"""Run repeated SEED RL matrix evaluations and gate the aggregate result.

Single matrix runs are noisy enough to produce false positives.  This workflow
evaluates each candidate across multiple seeds, aggregates each opponent row,
then runs the normal matrix gate on the aggregate matrix.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
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


def _expand_patterns(patterns: Iterable[str]) -> list[str]:
    checkpoints: list[str] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if not matches and Path(pattern + ".actor.weights.h5").exists():
            matches = [pattern + ".actor.weights.h5"]
        for match in matches:
            base = _checkpoint_base(match)
            if base not in checkpoints:
                checkpoints.append(base)
    return sorted(checkpoints)


def _read_opponents_from_jsonl(paths: Iterable[Path]) -> list[str]:
    opponents: list[str] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                opponent = row.get("opponent")
                if not opponent:
                    continue
                base = _checkpoint_base(str(opponent))
                if base not in opponents:
                    opponents.append(base)
    return opponents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--candidate-glob", action="append", default=[])
    parser.add_argument("--opponent", action="append", default=[])
    parser.add_argument("--opponent-glob", action="append", default=[])
    parser.add_argument("--opponent-file", action="append", type=Path, default=[])
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--seed-stride", type=int, default=100000)
    parser.add_argument("--hidden-dims", default="2048,2048,1024,512")
    parser.add_argument("--expected-count", type=int, default=13)
    parser.add_argument("--min-avg", type=float, default=103.9)
    parser.add_argument("--min-avg-lcb", type=float)
    parser.add_argument("--min-worst", type=float, default=-360.1)
    parser.add_argument("--max-row-std", type=float)
    parser.add_argument("--fail-fast-worst", type=float, default=-700.0)
    parser.add_argument("--max-invalid", type=float, default=0.001)
    parser.add_argument("--max-truncated", type=float, default=0.005)
    parser.add_argument("--max-row-invalid", type=float)
    parser.add_argument("--max-row-truncated", type=float)
    parser.add_argument("--max-opponent-regression", type=float, default=250.0)
    parser.add_argument("--no-baseline-improvement-required", action="store_true")
    parser.add_argument("--gate-pass-exit-code", type=int, default=0)
    parser.add_argument("--gate-fail-exit-code", type=int, default=1)
    return parser.parse_args()


def _default_out_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "runs" / f"seed_repeat_eval_{stamp}"


def _run(cmd: list[str]) -> int:
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT).returncode


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.pstdev(values)


def _aggregate_candidate(
    candidate: str,
    raw_paths: list[Path],
    out_path: Path,
) -> list[dict]:
    rows_by_opponent: dict[str, list[dict]] = {}
    for path in raw_paths:
        for row in _read_jsonl(path):
            name = str(row.get("opponent_name") or Path(str(row.get("opponent", ""))).name)
            rows_by_opponent.setdefault(name, []).append(row)

    aggregate_rows: list[dict] = []
    for opponent_name, rows in sorted(rows_by_opponent.items()):
        scores = [float(row.get("bb_per_100", 0.0)) for row in rows]
        invalids = [float(row.get("invalid_bet_rate", 0.0)) for row in rows]
        truncs = [float(row.get("truncated_hand_rate", 0.0)) for row in rows]
        first = rows[0]
        mean_score = _mean(scores)
        score_std = _stdev(scores)
        score_sem = score_std / (len(scores) ** 0.5) if scores else 0.0
        aggregate_rows.append({
            "model": candidate,
            "opponent": first.get("opponent"),
            "opponent_name": opponent_name,
            "bb_per_100": mean_score,
            "bb_per_100_min": min(scores),
            "bb_per_100_max": max(scores),
            "bb_per_100_std": score_std,
            "bb_per_100_sem": score_sem,
            "bb_per_100_lcb95": mean_score - 1.96 * score_sem,
            "bb_per_100_ucb95": mean_score + 1.96 * score_sem,
            "invalid_bet_rate": max(invalids),
            "invalid_bet_rate_avg": _mean(invalids),
            "truncated_hand_rate": max(truncs),
            "truncated_hand_rate_avg": _mean(truncs),
            "repeats": len(rows),
            "total_games": sum(int(row.get("total_games", 0)) for row in rows),
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in aggregate_rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return aggregate_rows


def _write_repeat_report(path: Path, aggregate_paths: list[Path]) -> None:
    lines = [
        "# SEED RL Repeat Evaluation",
        "",
        "| model | opponent | mean bb/100 | lcb95 | min repeat | max repeat | std | max invalid | max trunc | repeats |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for aggregate_path in aggregate_paths:
        for row in sorted(_read_jsonl(aggregate_path), key=lambda item: float(item.get("bb_per_100", 0.0))):
            lines.append(
                "| "
                f"{Path(str(row.get('model', aggregate_path.stem))).name} | "
                f"{row.get('opponent_name')} | "
                f"{float(row.get('bb_per_100', 0.0)):.1f} | "
                f"{float(row.get('bb_per_100_lcb95', row.get('bb_per_100', 0.0))):.1f} | "
                f"{float(row.get('bb_per_100_min', 0.0)):.1f} | "
                f"{float(row.get('bb_per_100_max', 0.0)):.1f} | "
                f"{float(row.get('bb_per_100_std', 0.0)):.1f} | "
                f"{float(row.get('invalid_bet_rate', 0.0)):.2%} | "
                f"{float(row.get('truncated_hand_rate', 0.0)):.2%} | "
                f"{int(row.get('repeats', 0))} |"
            )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")

    candidates = _expand_patterns(list(args.candidate_glob) + list(args.candidate))
    opponents = _expand_patterns(list(args.opponent_glob) + list(args.opponent))
    for opponent in _read_opponents_from_jsonl(args.opponent_file):
        if opponent not in opponents:
            opponents.append(opponent)
    if not candidates:
        raise SystemExit("No candidate checkpoints found")
    if not opponents:
        raise SystemExit("No opponent checkpoints found")

    out_dir = args.out_dir or _default_out_dir()
    raw_dir = out_dir / "raw"
    aggregate_dir = out_dir / "aggregate"
    raw_dir.mkdir(parents=True, exist_ok=True)
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    statuses: list[dict[str, object]] = []
    aggregate_paths: list[Path] = []
    hard_failure = False

    for candidate_idx, candidate in enumerate(candidates):
        raw_paths: list[Path] = []
        label = Path(candidate).name
        for repeat_idx in range(args.repeats):
            seed = args.seed + candidate_idx * args.seed_stride * args.repeats + repeat_idx * args.seed_stride
            matrix_path = raw_dir / f"{label}_repeat{repeat_idx + 1}_seed{seed}.jsonl"
            cmd = [
                sys.executable,
                str(SCRIPT_DIR / "evaluate_seed_pool.py"),
                "--model",
                candidate,
                "--games",
                str(args.games),
                "--seed",
                str(seed),
                "--hidden-dims",
                args.hidden_dims,
                "--out",
                str(matrix_path),
                "--fail-fast-worst-bb100",
                str(args.fail_fast_worst),
                "--max-invalid-rate",
                str(args.max_invalid),
                "--max-truncated-rate",
                str(args.max_truncated),
                "--fail-exit-code",
                "3",
                "--continue-on-gate-fail",
            ]
            for opponent in opponents:
                cmd.extend(["--opponent", opponent])
            rc = _run(cmd)
            if rc not in (0, 3):
                hard_failure = True
            raw_paths.append(matrix_path)
            statuses.append({
                "candidate": candidate,
                "repeat": repeat_idx + 1,
                "seed": seed,
                "matrix": str(matrix_path),
                "return_code": rc,
            })

        aggregate_path = aggregate_dir / f"{label}.jsonl"
        _aggregate_candidate(candidate, raw_paths, aggregate_path)
        aggregate_paths.append(aggregate_path)

    (out_dir / "repeat_status.json").write_text(
        json.dumps(statuses, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_repeat_report(out_dir / "repeat_report.md", aggregate_paths)

    gate_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "seed_matrix_gate.py"),
        *[str(path) for path in aggregate_paths],
        "--out",
        str(out_dir / "gate_report.md"),
        "--expected-count",
        str(args.expected_count),
        "--min-avg",
        str(args.min_avg),
        "--min-worst",
        str(args.min_worst),
        "--fail-fast-worst",
        str(args.fail_fast_worst),
        "--max-invalid",
        str(args.max_invalid),
        "--max-truncated",
        str(args.max_truncated),
        "--pass-exit-code",
        str(args.gate_pass_exit_code),
        "--fail-exit-code",
        str(args.gate_fail_exit_code),
    ]
    if args.max_row_invalid is not None:
        gate_cmd.extend(["--max-row-invalid", str(args.max_row_invalid)])
    if args.max_row_truncated is not None:
        gate_cmd.extend(["--max-row-truncated", str(args.max_row_truncated)])
    if args.min_avg_lcb is not None:
        gate_cmd.extend(["--min-avg-lcb", str(args.min_avg_lcb)])
    if args.max_row_std is not None:
        gate_cmd.extend(["--max-row-std", str(args.max_row_std)])
    if args.baseline:
        gate_cmd.extend(["--baseline", str(args.baseline)])
        if not args.no_baseline_improvement_required:
            gate_cmd.append("--require-baseline-improvement")
        if args.max_opponent_regression is not None:
            gate_cmd.extend(["--max-opponent-regression", str(args.max_opponent_regression)])

    gate_rc = _run(gate_cmd)
    print(f"Wrote {out_dir / 'gate_report.md'}", flush=True)
    print(f"Wrote {out_dir / 'repeat_report.md'}", flush=True)
    if hard_failure:
        return 2
    return gate_rc


if __name__ == "__main__":
    raise SystemExit(main())
