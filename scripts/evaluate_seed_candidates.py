#!/usr/bin/env python3
"""Evaluate multiple SEED RL checkpoints and run the retention gate."""

from __future__ import annotations

import argparse
import glob
import json
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
    parser.add_argument(
        "--opponent-file",
        action="append",
        type=Path,
        default=[],
        help="Matrix JSONL file(s) whose opponent paths should form the opponent pool",
    )
    parser.add_argument("--baseline", type=Path, help="Baseline matrix JSONL")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--hidden-dims", default="2048,2048,1024,512")
    parser.add_argument("--expected-count", type=int, default=13)
    parser.add_argument("--min-avg", type=float, default=103.9)
    parser.add_argument("--min-worst", type=float, default=-360.1)
    parser.add_argument("--fail-fast-worst", type=float, default=-500.0)
    parser.add_argument("--max-invalid", type=float, default=0.001)
    parser.add_argument("--max-truncated", type=float, default=0.005)
    parser.add_argument("--max-row-invalid", type=float)
    parser.add_argument("--max-row-truncated", type=float)
    parser.add_argument(
        "--max-opponent-regression",
        type=float,
        default=250.0,
        help="With --baseline, fail if a matched opponent regresses by more than this bb/100",
    )
    parser.add_argument(
        "--no-baseline-improvement-required",
        action="store_true",
        help="Do not require candidate avg/worst to improve over --baseline",
    )
    parser.add_argument(
        "--continue-on-row-gate-fail",
        action="store_true",
        help="Finish full matrices even when a row gate fails",
    )
    parser.add_argument(
        "--disable-row-gates",
        action="store_true",
        help="Do not pass per-row fail-fast gates into evaluate_seed_pool.py",
    )
    parser.add_argument("--row-fail-exit-code", type=int, default=3)
    parser.add_argument("--gate-pass-exit-code", type=int, default=0)
    parser.add_argument("--gate-fail-exit-code", type=int, default=1)
    return parser.parse_args()


def _default_out_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "runs" / f"seed_candidate_eval_{stamp}"


def _run(cmd: list[str]) -> int:
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT).returncode


def main() -> int:
    args = parse_args()
    candidate_patterns = list(args.candidate_glob) + list(args.candidate)
    opponent_patterns = list(args.opponent_glob) + list(args.opponent)
    candidates = _expand_patterns(candidate_patterns)
    opponents = _expand_patterns(opponent_patterns)
    for opponent in _read_opponents_from_jsonl(args.opponent_file):
        if opponent not in opponents:
            opponents.append(opponent)
    if not candidates:
        raise SystemExit("No candidate checkpoints found")
    if not opponents:
        raise SystemExit("No opponent checkpoints found")

    out_dir = args.out_dir or _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    matrices: list[Path] = []
    statuses: list[dict[str, object]] = []
    hard_failure = False

    for idx, candidate in enumerate(candidates):
        matrix_path = out_dir / f"{Path(candidate).name}.jsonl"
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "evaluate_seed_pool.py"),
            "--model",
            candidate,
            "--games",
            str(args.games),
            "--seed",
            str(args.seed + idx * 1000),
            "--hidden-dims",
            args.hidden_dims,
            "--out",
            str(matrix_path),
        ]
        for opponent in opponents:
            cmd.extend(["--opponent", opponent])
        if not args.disable_row_gates:
            cmd.extend([
                "--fail-fast-worst-bb100",
                str(args.fail_fast_worst),
                "--max-invalid-rate",
                str(args.max_invalid),
                "--max-truncated-rate",
                str(args.max_truncated),
                "--fail-exit-code",
                str(args.row_fail_exit_code),
            ])
        if args.continue_on_row_gate_fail:
            cmd.append("--continue-on-gate-fail")

        rc = _run(cmd)
        has_matrix = matrix_path.exists() and matrix_path.stat().st_size > 0
        if has_matrix:
            matrices.append(matrix_path)
        if rc not in (0, args.row_fail_exit_code):
            hard_failure = True
        statuses.append({
            "candidate": candidate,
            "matrix": str(matrix_path) if has_matrix else None,
            "return_code": rc,
        })

    status_path = out_dir / "candidate_status.json"
    status_path.write_text(
        json.dumps(statuses, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if not matrices:
        print(f"No matrix rows were produced. See {status_path}", file=sys.stderr)
        return 2

    gate_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "seed_matrix_gate.py"),
        *[str(path) for path in matrices],
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
    if args.baseline:
        gate_cmd.extend(["--baseline", str(args.baseline)])
        if not args.no_baseline_improvement_required:
            gate_cmd.append("--require-baseline-improvement")
        if args.max_opponent_regression is not None:
            gate_cmd.extend([
                "--max-opponent-regression",
                str(args.max_opponent_regression),
            ])

    gate_rc = _run(gate_cmd)
    print(f"Wrote {out_dir / 'gate_report.md'}", flush=True)
    if hard_failure:
        return 2
    return gate_rc


if __name__ == "__main__":
    raise SystemExit(main())
