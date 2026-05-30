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


def raise_bucket_count() -> int:
    return len(PREFLOP_RAISE_MULTIPLIERS)

