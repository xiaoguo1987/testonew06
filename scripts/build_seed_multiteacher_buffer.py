#!/usr/bin/env python3
"""Collect row-balanced state/probability buffers from multiple teacher policies."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_seed_reference_buffer import _checkpoint_base, _play_reference_hands


def _load_pairs(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    return list(data.get("selected", []))


def _parse_pair(raw: str) -> dict:
    # Accepted format: row=teacher=opponent or teacher=opponent.
    parts = raw.split("=")
    if len(parts) == 2:
        teacher, opponent = parts
        row = Path(_checkpoint_base(opponent)).name
    elif len(parts) == 3:
        row, teacher, opponent = parts
    else:
        raise SystemExit(f"--pair must be teacher=opponent or row=teacher=opponent, got: {raw}")
    return {"row": row, "teacher": teacher, "opponent": opponent}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-file", action="append", type=Path, default=[])
    parser.add_argument("--pair", action="append", default=[])
    parser.add_argument("--states-per-row", type=int, default=5000)
    parser.add_argument("--max-hands-per-row", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--seed-step", type=int, default=997)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--hidden-dims", default="2048,2048,1024,512")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from poker_ai.config.ppo_config import PPOConfig
    from poker_ai.model.large_ppo_model import LargePPOModel
    from poker_ai.training.gpu_trainer import SeedRLTrainer

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    pairs: list[dict] = []
    for path in args.selection_file:
        pairs.extend(_load_pairs(path))
    for raw in args.pair:
        pairs.append(_parse_pair(raw))
    if not pairs:
        raise SystemExit("No teacher/opponent pairs provided")

    hidden_dims = [int(part) for part in args.hidden_dims.split(",") if part.strip()]
    states: list[np.ndarray] = []
    target_probs: list[np.ndarray] = []
    row_ids: list[int] = []
    row_names: list[str] = []
    summaries: list[dict] = []

    for idx, spec in enumerate(pairs):
        teacher = _checkpoint_base(str(spec.get("teacher") or spec.get("model") or ""))
        opponent = _checkpoint_base(str(spec.get("opponent") or ""))
        if not teacher or not opponent:
            raise SystemExit(f"Invalid pair: {spec}")
        if not Path(teacher + ".actor.weights.h5").exists():
            raise SystemExit(f"Missing teacher actor weights: {teacher}")
        if not Path(opponent + ".actor.weights.h5").exists():
            raise SystemExit(f"Missing opponent actor weights: {opponent}")

        row_name = str(spec.get("row") or spec.get("opponent_name") or Path(opponent).name)
        row_names.append(row_name)
        cfg = PPOConfig(hidden_dims=hidden_dims)
        model = LargePPOModel(hidden_dims=hidden_dims, config=cfg)
        trainer = SeedRLTrainer(
            model=model,
            checkpoint_dir=str(Path(teacher).parent),
            num_eval_games=0,
            curriculum_start_phase=2,
        )
        if not trainer.load_checkpoint(teacher):
            raise SystemExit(f"Failed to load teacher checkpoint: {teacher}")

        before = len(states)
        logging.info(
            "Collecting row=%s teacher=%s opponent=%s (%s/%s)",
            row_name,
            Path(teacher).name,
            Path(opponent).name,
            idx + 1,
            len(pairs),
        )
        summary = _play_reference_hands(
            trainer=trainer,
            opponent=opponent,
            states=states,
            target_probs=target_probs,
            states_per_opponent=max(1, int(args.states_per_row)),
            max_hands=max(1, int(args.max_hands_per_row)),
            seed=int(args.seed) + idx * int(args.seed_step),
            temperature=max(float(args.temperature), 1e-3),
            epsilon=max(0.0, min(float(args.epsilon), 1.0)),
        )
        added = len(states) - before
        row_ids.extend([idx] * added)
        summary.update(
            {
                "row": row_name,
                "row_id": idx,
                "teacher": teacher,
                "teacher_name": Path(teacher).name,
                "teacher_score": spec.get("teacher_score"),
            }
        )
        summaries.append(summary)

    if not states:
        raise SystemExit("No states collected")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "pairs": summaries,
        "rows": row_names,
        "states": len(states),
        "states_per_row": int(args.states_per_row),
        "temperature": float(args.temperature),
        "epsilon": float(args.epsilon),
        "seed": int(args.seed),
    }
    np.savez_compressed(
        out_path,
        states=np.asarray(states, dtype=np.float32),
        target_probs=np.asarray(target_probs, dtype=np.float32),
        row_ids=np.asarray(row_ids, dtype=np.int32),
        row_names=np.asarray(row_names, dtype="U"),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True), dtype="U"),
    )
    print(json.dumps({"out": str(out_path), **metadata}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
