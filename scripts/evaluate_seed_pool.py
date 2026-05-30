#!/usr/bin/env python3
"""Evaluate one SEED RL checkpoint against a list of historical checkpoints."""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from pathlib import Path
from typing import Iterable

from poker_ai.model.large_ppo_model import LargePPOModel
from poker_ai.training.gpu_trainer import SeedRLTrainer


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
    parser.add_argument("--out", default=None, help="Optional JSONL output path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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

    model = LargePPOModel()
    trainer = SeedRLTrainer(
        model=model,
        checkpoint_dir=str(Path(model_path).parent),
        num_eval_games=args.games,
        curriculum_start_phase=2,
    )
    if not trainer.load_checkpoint(model_path):
        raise SystemExit(f"Failed to load model checkpoint: {model_path}")

    out_fh = open(args.out, "w", encoding="utf-8") if args.out else None
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
            text = json.dumps(row, ensure_ascii=False, sort_keys=True)
            print(text, flush=True)
            if out_fh:
                out_fh.write(text + "\n")
                out_fh.flush()
    finally:
        if out_fh:
            out_fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
