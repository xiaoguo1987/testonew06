#!/usr/bin/env python3
"""Export SEED-RL experiment JSON config values as shell assignments."""

from __future__ import annotations

import argparse
import json
import os
import shlex
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_absolute():
        path = ROOT / path
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise SystemExit(f"Config must be a JSON object: {path}")
    return data


def _set(mapping: dict[str, str], name: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        mapping[name] = "1" if value else "0"
        return
    if isinstance(value, (list, tuple)):
        mapping[name] = " ".join(str(item) for item in value)
        return
    mapping[name] = str(value)


def _reference_smoke_exports(config: dict[str, Any]) -> dict[str, str]:
    baseline = config.get("baseline", {})
    opponents = config.get("opponents", {})
    paths = config.get("paths", {})
    smoke = config.get("reference_smoke", {})

    exports: dict[str, str] = {}
    _set(exports, "SEED_EXPERIMENT_ID", config.get("id"))
    _set(exports, "ANCHOR_NAME", baseline.get("checkpoint_name"))
    _set(exports, "ANCHOR_HINT", baseline.get("checkpoint_hint"))
    _set(exports, "ARCHIVE_DIR", paths.get("archive_dir"))
    _set(exports, "RUN_NAME_PREFIX", smoke.get("run_name_prefix"))
    _set(exports, "NUM_HANDS", smoke.get("num_hands"))
    _set(exports, "LEARNING_RATE", smoke.get("learning_rate"))
    _set(exports, "PPO_CLIP", smoke.get("ppo_clip"))
    _set(exports, "PPO_ENTROPY_COEF", smoke.get("ppo_entropy_coef"))
    _set(exports, "ANCHOR_KL_COEF", smoke.get("anchor_kl_coef"))
    _set(exports, "MAX_ANCHOR_KL", smoke.get("max_anchor_kl"))
    _set(exports, "REF_COEF", smoke.get("reference_distill_coef"))
    _set(exports, "REF_BATCH_SIZE", smoke.get("reference_batch_size"))
    _set(exports, "MAX_REFERENCE_KL", smoke.get("max_reference_kl"))
    _set(exports, "MAX_REFERENCE_MEAN_KL", smoke.get("max_reference_mean_kl"))
    _set(exports, "MAX_REFERENCE_P95_KL", smoke.get("max_reference_p95_kl"))
    _set(exports, "VALIDATION_GAMES", smoke.get("validation_games"))
    _set(exports, "VALIDATION_REPEATS", smoke.get("validation_repeats"))
    _set(exports, "VALIDATION_MIN_AVG", smoke.get("validation_min_avg"))
    _set(exports, "VALIDATION_MIN_WORST", smoke.get("validation_min_worst"))
    _set(exports, "VALIDATION_MIN_ROW_MIN", smoke.get("validation_min_row_min"))
    _set(exports, "VALIDATION_MAX_ROW_STD", smoke.get("validation_max_row_std"))
    _set(exports, "VALIDATION_FAIL_FAST_WORST", smoke.get("validation_fail_fast_worst"))
    _set(exports, "VALIDATION_MAX_INVALID", smoke.get("validation_max_invalid"))
    _set(exports, "VALIDATION_MAX_TRUNCATED", smoke.get("validation_max_truncated"))
    _set(exports, "OPPONENT_NAMES", opponents.get("hard_subset"))
    _set(exports, "OPPONENT_ROSTER_FILE", opponents.get("roster_file"))
    return exports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--scope",
        choices=["reference-smoke"],
        default="reference-smoke",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep values already present in the process environment.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = _load_config(args.config)
    exports = _reference_smoke_exports(config)
    for name in sorted(exports):
        if args.skip_existing and name in os.environ:
            continue
        print(f"{name}={shlex.quote(exports[name])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
