# -*- coding: utf-8 -*-
"""Shared abstract action space for NLHE training.

The seed-rl path uses a 12-action abstraction:
fold, call/check, nine raise sizes, and all-in.  Preflop raise buckets are
raise-to multipliers of the current bet/blind; postflop buckets are raise-to
amounts of current bet plus a fraction of the pot.
"""

from __future__ import annotations

from typing import List


ACTION_NAMES: List[str] = [
    "fold",
    "call",
    "raise_25pct",
    "raise_33pct",
    "raise_50pct",
    "raise_66pct",
    "raise_75pct",
    "raise_100pct",
    "raise_125pct",
    "raise_150pct",
    "raise_200pct",
    "allin",
]

ACTION_DIM = len(ACTION_NAMES)
ALLIN_ACTION_INDEX = ACTION_DIM - 1
RAISE_ACTION_START = 2
RAISE_ACTION_END = ACTION_DIM - 2

PREFLOP_RAISE_MULTIPLIERS = [2.25, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0]
POSTFLOP_POT_FRACTIONS = [0.25, 0.33, 0.50, 0.66, 0.75, 1.00, 1.25, 1.50, 2.00]

# All-in is a tactical low-SPR / short-stack action, not a normal 100bb raise
# bucket.  Keeping it gated in the legal mask prevents the policy from learning
# deep-stack open-shove shortcuts while still allowing commitment spots.
ALLIN_SHORT_STACK_BB = 25.0
ALLIN_POSTFLOP_MAX_SPR = 1.5
ALLIN_COMMIT_STACK_FRACTION = 0.35
ALLIN_COMMIT_TOTAL_FRACTION = 0.25


def raise_bucket_count() -> int:
    return len(PREFLOP_RAISE_MULTIPLIERS)


def is_allin_bucket_allowed(
    round_num: int,
    pot: int,
    current_bet: int,
    big_blind: int,
    player_bet: int,
    player_stack: int,
) -> bool:
    """Return whether the abstract all-in bucket should be exposed.

    This is stricter than poker legality.  The engine can still process an
    all-in call or an all-in caused by another raise bucket being capped by the
    player's stack; this only controls whether the explicit all-in action is
    available to the policy.
    """
    bb = max(float(big_blind), 1.0)
    stack = max(float(player_stack), 0.0)
    if stack <= 0:
        return False

    total_stack = max(float(player_bet + player_stack), bb)
    call_amount = max(0.0, float(current_bet - player_bet))
    pot_after_call = max(float(pot) + call_amount, bb)
    stack_bb = stack / bb
    pressure = call_amount / stack if stack > 0 else 0.0
    committed = float(current_bet) / total_stack

    if stack_bb <= ALLIN_SHORT_STACK_BB:
        return True
    if pressure >= ALLIN_COMMIT_STACK_FRACTION or committed >= ALLIN_COMMIT_TOTAL_FRACTION:
        return True
    if int(round_num) > 1 and (stack / pot_after_call) <= ALLIN_POSTFLOP_MAX_SPR:
        return True
    return False
