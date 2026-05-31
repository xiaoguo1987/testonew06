#!/usr/bin/env python3
"""Evaluate one SEED RL checkpoint against a list of historical checkpoints."""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Main checkpoint base path")
    parser.add_argument(
        "--opponent",
        action="append",
        default=[],
        help="Opponent checkpoint base path or .actor.weights.h5 file",
    )
    parser.add_argument(
        "--opponent-glob",
        action="append",
        default=[],
        help="Glob for opponent .actor.weights.h5 files",
    )
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument(
        "--hidden-dims",
        default="2048,2048,1024,512",
        help="Comma-separated LargePPO hidden dimensions",
    )
    parser.add_argument("--out", default=None, help="Optional JSONL output path")
    parser.add_argument(
        "--fail-fast-worst-bb100",
        type=float,
        default=None,
        help="Stop evaluation if any opponent result drops below this bb/100",
    )
    parser.add_argument(
        "--max-invalid-rate",
        type=float,
        default=None,
        help="Stop evaluation if any opponent result exceeds this invalid bet rate",
    )
    parser.add_argument(
        "--max-truncated-rate",
        type=float,
        default=None,
        help="Stop evaluation if any opponent result exceeds this truncated hand rate",
    )
    parser.add_argument(
        "--continue-on-gate-fail",
        action="store_true",
        help="Tag gate failures but continue evaluating the remaining opponents",
    )
    parser.add_argument(
        "--fail-exit-code",
        type=int,
        default=3,
        help="Exit code when a fail-fast evaluation gate is triggered",
    )
    return parser.parse_args()


def _result_gate_failures(row: dict, args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    bb100 = float(row.get("bb_per_100", 0.0))
    invalid = float(row.get("invalid_bet_rate", 0.0))
    truncated = float(row.get("truncated_hand_rate", 0.0))

    if (
        args.fail_fast_worst_bb100 is not None
        and bb100 < args.fail_fast_worst_bb100
    ):
        failures.append(
            f"bb_per_100 {bb100:.1f} < {args.fail_fast_worst_bb100:.1f}"
        )
    if args.max_invalid_rate is not None and invalid > args.max_invalid_rate:
        failures.append(
            f"invalid_bet_rate {invalid:.4f} > {args.max_invalid_rate:.4f}"
        )
    if args.max_truncated_rate is not None and truncated > args.max_truncated_rate:
        failures.append(
            f"truncated_hand_rate {truncated:.4f} > {args.max_truncated_rate:.4f}"
        )
    return failures


def main() -> int:
    args = parse_args()
    from poker_ai.config.ppo_config import PPOConfig
    from poker_ai.model.large_ppo_model import LargePPOModel
    from poker_ai.training.gpu_trainer import SeedRLTrainer

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    opponent_patterns = list(args.opponent_glob)
    opponent_patterns.extend(args.opponent)
    opponents = _expand_patterns(opponent_patterns)
    model_path = _checkpoint_base(args.model)
    opponents = [
        opp for opp in opponents
        if os.path.abspath(opp) != os.path.abspath(model_path)
    ]
    if not opponents:
        raise SystemExit("No opponent checkpoints found")

    hidden_dims = [int(part) for part in args.hidden_dims.split(",") if part.strip()]
    cfg = PPOConfig(hidden_dims=hidden_dims)
    model = LargePPOModel(hidden_dims=hidden_dims, config=cfg)
    trainer = SeedRLTrainer(
        model=model,
        checkpoint_dir=str(Path(model_path).parent),
        num_eval_games=args.games,
        curriculum_start_phase=2,
    )
    if not trainer.load_checkpoint(model_path):
        raise SystemExit(f"Failed to load model checkpoint: {model_path}")

    out_fh = open(args.out, "w", encoding="utf-8") if args.out else None
    exit_code = 0
    try:
        for idx, opponent in enumerate(opponents):
            trainer.opp_model.load_weights(opponent)
            result = trainer._run_evaluation(
                args.games,
                opponent_policy="opponent_model",
                seed=args.seed + idx,
            )
            row = {
                "model": model_path,
                "opponent": opponent,
                "opponent_name": os.path.basename(opponent),
                **result,
            }
            failures = _result_gate_failures(row, args)
            row["gate_status"] = "FAIL" if failures else "OK"
            if failures:
                row["gate_failures"] = failures
            text = json.dumps(row, ensure_ascii=False, sort_keys=True)
            print(text, flush=True)
            if out_fh:
                out_fh.write(text + "\n")
                out_fh.flush()
            if failures and not args.continue_on_gate_fail:
                logging.error(
                    "Evaluation gate failed for %s: %s",
                    row["opponent_name"],
                    "; ".join(failures),
                )
                exit_code = args.fail_exit_code
                break
    finally:
        if out_fh:
            out_fh.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
