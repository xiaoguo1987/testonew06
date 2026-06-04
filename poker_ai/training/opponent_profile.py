"""Opponent behavior profile tracking shared by SEED-RL train/eval paths."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


OpponentHandFlags = List[Dict[str, float]]


def new_opponent_hand_flags(num_players: int = 6) -> OpponentHandFlags:
    """Create per-hand opponent observation flags.

    Stats are updated once per finished hand, keeping the online EMA stable and
    making training and evaluation use the same interpretation of VPIP/PFR/AF.
    """
    return [
        {
            "seen": 0.0,
            "vpip": 0.0,
            "pfr": 0.0,
            "agg_actions": 0.0,
            "passive_actions": 0.0,
            "faced_3bet": 0.0,
            "fold_to_3bet": 0.0,
        }
        for _ in range(num_players)
    ]


def observe_profile_action(
    flags: OpponentHandFlags,
    player_index: int,
    action: str,
    amount: Optional[int],
    round_num: int,
    preflop_raise_count: int,
) -> int:
    """Observe an executed action and return the updated preflop raise count."""
    raises_before = preflop_raise_count
    if round_num == 1 and action == "raise":
        preflop_raise_count += 1

    if player_index <= 0 or player_index >= len(flags):
        return preflop_raise_count

    row = flags[player_index]
    row["seen"] = 1.0
    paid = float(amount or 0) > 0.0
    if round_num == 1:
        if action in ("call", "raise") and paid:
            row["vpip"] = 1.0
        if action == "raise":
            row["pfr"] = 1.0
        if raises_before >= 2:
            row["faced_3bet"] = 1.0
            if action == "fold":
                row["fold_to_3bet"] = 1.0
    if action == "raise":
        row["agg_actions"] += 1.0
    elif action == "call" and paid:
        row["passive_actions"] += 1.0
    return preflop_raise_count


def update_opponent_profile_ema(
    stats: np.ndarray,
    flags: OpponentHandFlags,
    decay: float = 0.99,
) -> None:
    """Apply per-hand flags to `[seat, VPIP/PFR/AF/fold3bet]` EMA stats."""
    max_players = min(len(flags), int(stats.shape[0]))
    for pi in range(1, max_players):
        row = flags[pi]
        if row.get("seen", 0.0) <= 0.0:
            continue
        target = stats[pi]
        denom = row.get("agg_actions", 0.0) + row.get("passive_actions", 0.0)
        agg_freq = row.get("agg_actions", 0.0) / denom if denom > 0.0 else 0.0
        obs = np.array(
            [row.get("vpip", 0.0), row.get("pfr", 0.0), agg_freq, target[3]],
            dtype=np.float32,
        )
        if row.get("faced_3bet", 0.0) > 0.0:
            obs[3] = row.get("fold_to_3bet", 0.0)
        target[:] = decay * target + (1.0 - decay) * obs
