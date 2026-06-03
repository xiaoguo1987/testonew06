#!/usr/bin/env python3
"""Build a fixed policy-distillation reference buffer for SEED RL repair runs.

The buffer stores Main_AI decision states plus the frozen reference policy's
full action distribution.  During PPO repair, LargePPOModel can sample these
states every mini-batch and penalize KL(target || current), preventing the
policy from forgetting hard-opponent states that are not present in the current
rollout buffer.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

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


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _opponent_name(row: dict) -> str:
    raw = row.get("opponent_name") or row.get("opponent") or "unknown"
    return os.path.basename(str(raw).replace(".actor.weights.h5", ""))


def _load_opponents(args: argparse.Namespace) -> list[str]:
    opponents = _expand_patterns(list(args.opponent_glob) + list(args.opponent))
    scored_rows: list[dict] = []
    for path in args.opponent_file:
        scored_rows.extend(_read_jsonl(path))
    scored_rows.sort(key=lambda row: float(row.get("bb_per_100", 0.0)))

    for row in scored_rows[: max(0, args.worst_count)]:
        opponent = row.get("opponent")
        if opponent:
            opponents.append(_checkpoint_base(str(opponent)))

    if args.always_include:
        wanted = {
            os.path.basename(str(name).replace(".actor.weights.h5", ""))
            for name in args.always_include
        }
        for row in scored_rows:
            if _opponent_name(row) in wanted and row.get("opponent"):
                opponents.append(_checkpoint_base(str(row["opponent"])))

    out: list[str] = []
    seen: set[str] = set()
    model_abs = os.path.abspath(_checkpoint_base(args.model))
    for opponent in opponents:
        base = _checkpoint_base(opponent)
        key = os.path.abspath(base)
        if key == model_abs or key in seen:
            continue
        if not Path(base + ".actor.weights.h5").exists():
            logging.warning("Skipping missing opponent weights: %s", base)
            continue
        seen.add(key)
        out.append(base)
    return out


def _normalize_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float32)
    probs = np.clip(probs, 1e-6, 1.0)
    total = float(probs.sum())
    if total <= 1e-8 or not np.isfinite(total):
        return np.full_like(probs, 1.0 / len(probs), dtype=np.float32)
    return (probs / total).astype(np.float32, copy=False)


def _play_reference_hands(
    trainer,
    opponent: str,
    states: list[np.ndarray],
    target_probs: list[np.ndarray],
    states_per_opponent: int,
    max_hands: int,
    seed: int,
    temperature: float,
    epsilon: float,
) -> dict:
    from poker_ai.action_space import ACTION_DIM
    from poker_ai.game.poker_game import Player, PokerGame
    from poker_ai.model.state_representation import StateRepresentation
    from poker_ai.training.env_worker import _legal_mask

    trainer.opp_model.load_weights(opponent)
    rng_state = np.random.get_state()
    np.random.seed(int(seed))
    sr = StateRepresentation()

    hands = 0
    decisions = 0
    truncated_hands = 0
    invalid_actions = 0
    max_actions_per_hand = 256
    start_rows = len(states)

    try:
        while decisions < states_per_opponent and hands < max_hands:
            players = [Player("Main_AI", trainer.initial_stack, True)]
            for i in range(5):
                players.append(Player(f"Opponent_{i + 1}", trainer.initial_stack, True))
            game = PokerGame(trainer.small_blind, trainer.big_blind, players)
            game.dealer_pos = np.random.randint(6)
            game.start_new_hand()
            hands += 1

            hand_actions = 0
            while not game.is_hand_over():
                hand_actions += 1
                if hand_actions > max_actions_per_hand:
                    truncated_hands += 1
                    players[0].is_active = False
                    alive = [p for p in players if p.is_active]
                    if alive and game.pot > 0:
                        game._award_pot(alive[0])
                    break

                if game.is_round_over():
                    if game.proceed_to_next_round():
                        break
                cp = game.next_player()
                if not cp or not cp.is_active:
                    continue

                state = sr.get_state(game, cp)
                if cp.name == "Main_AI":
                    action_probs, _ = trainer.model.predict(state)
                    states.append(np.asarray(state, dtype=np.float32))
                    target_probs.append(_normalize_probs(action_probs))
                    decisions += 1
                    legal_mask = _legal_mask(game, cp)
                    action_idx = trainer._select_eval_action(
                        action_probs,
                        legal_mask,
                        stochastic=True,
                        temperature=temperature,
                        epsilon=epsilon,
                    )
                else:
                    action_probs, _ = trainer.opp_model.predict(state)
                    legal_mask = _legal_mask(game, cp)
                    action_idx = trainer._select_eval_action(
                        action_probs,
                        legal_mask,
                        stochastic=True,
                        temperature=temperature,
                        epsilon=epsilon,
                    )

                action_vec = np.zeros(ACTION_DIM, dtype=np.float32)
                action_vec[action_idx] = 1.0
                hand_strength = sr.calculate_hand_strength(cp, game)
                action, amount = sr.decode_action(action_vec, game, cp, hand_strength)
                if not game.handle_action(cp, action, amount):
                    if cp.name == "Main_AI":
                        invalid_actions += 1
                    game.handle_action(cp, "fold", None)

                if decisions >= states_per_opponent and cp.name == "Main_AI":
                    break

            alive = [p for p in players if p.is_active]
            if len(alive) == 1 and game.pot > 0:
                game._award_pot(alive[0])
    finally:
        np.random.set_state(rng_state)

    return {
        "opponent": opponent,
        "opponent_name": os.path.basename(opponent),
        "hands": hands,
        "states": len(states) - start_rows,
        "truncated_hands": truncated_hands,
        "invalid_actions": invalid_actions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Frozen reference checkpoint base path")
    parser.add_argument("--opponent", action="append", default=[])
    parser.add_argument("--opponent-glob", action="append", default=[])
    parser.add_argument("--opponent-file", action="append", type=Path, default=[])
    parser.add_argument("--always-include", action="append", default=[])
    parser.add_argument("--worst-count", type=int, default=6)
    parser.add_argument("--states-per-opponent", type=int, default=50000)
    parser.add_argument("--max-hands-per-opponent", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--seed-step", type=int, default=997)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--hidden-dims", default="2048,2048,1024,512")
    parser.add_argument("--out", required=True, help="Output .npz path")
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

    model_path = _checkpoint_base(args.model)
    opponents = _load_opponents(args)
    if not opponents:
        raise SystemExit("No opponent checkpoints found")

    hidden_dims = [int(part) for part in args.hidden_dims.split(",") if part.strip()]
    cfg = PPOConfig(hidden_dims=hidden_dims)
    model = LargePPOModel(hidden_dims=hidden_dims, config=cfg)
    trainer = SeedRLTrainer(
        model=model,
        checkpoint_dir=str(Path(model_path).parent),
        num_eval_games=0,
        curriculum_start_phase=2,
    )
    if not trainer.load_checkpoint(model_path):
        raise SystemExit(f"Failed to load model checkpoint: {model_path}")

    states: list[np.ndarray] = []
    target_probs: list[np.ndarray] = []
    summaries: list[dict] = []
    for idx, opponent in enumerate(opponents):
        logging.info(
            "Collecting reference states against %s (%s/%s)",
            os.path.basename(opponent),
            idx + 1,
            len(opponents),
        )
        summary = _play_reference_hands(
            trainer=trainer,
            opponent=opponent,
            states=states,
            target_probs=target_probs,
            states_per_opponent=max(1, int(args.states_per_opponent)),
            max_hands=max(1, int(args.max_hands_per_opponent)),
            seed=int(args.seed) + idx * int(args.seed_step),
            temperature=max(float(args.temperature), 1e-3),
            epsilon=max(0.0, min(float(args.epsilon), 1.0)),
        )
        summaries.append(summary)
        logging.info(
            "Collected %s states from %s hands against %s",
            summary["states"],
            summary["hands"],
            summary["opponent_name"],
        )

    if not states:
        raise SystemExit("No reference states collected")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": model_path,
        "opponents": summaries,
        "states": len(states),
        "states_per_opponent": int(args.states_per_opponent),
        "temperature": float(args.temperature),
        "epsilon": float(args.epsilon),
        "seed": int(args.seed),
    }
    np.savez_compressed(
        out_path,
        states=np.asarray(states, dtype=np.float32),
        target_probs=np.asarray(target_probs, dtype=np.float32),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True), dtype="U"),
    )
    print(json.dumps({"out": str(out_path), **metadata}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
