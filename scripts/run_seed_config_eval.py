#!/usr/bin/env python3
"""Run a config-driven SEED-RL repeated evaluation."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_absolute():
        path = ROOT / path
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise SystemExit(f"Config must be a JSON object: {path}")
    return data


def _checkpoint_base(path: str) -> str:
    if path.endswith(".actor.weights.h5"):
        return path[: -len(".actor.weights.h5")]
    return path


def _candidate_from_hint(root: Path, hint: str | None) -> str | None:
    if not hint:
        return None
    path = Path(hint)
    if not path.is_absolute():
        path = root / path
    base = _checkpoint_base(str(path))
    if Path(base + ".actor.weights.h5").exists():
        return base
    return None


def _resolve_checkpoint(root: Path, name: str, archive_dir: str | None) -> str:
    raw = _checkpoint_base(name)
    raw_path = Path(raw)
    if raw_path.is_absolute() and Path(raw + ".actor.weights.h5").exists():
        return raw
    if not raw_path.is_absolute() and Path(str(root / raw) + ".actor.weights.h5").exists():
        return str(root / raw)

    search_dirs = [root / "runs", root / "checkpoints"]
    if archive_dir:
        search_dirs.append(Path(archive_dir))
    target = f"{Path(raw).name}.actor.weights.h5"
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for match in search_dir.rglob(target):
            return _checkpoint_base(str(match))
    raise SystemExit(f"Unable to resolve checkpoint: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/seed_v0_baseline.json"))
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--pool", choices=["hard_subset", "full_pool"], default="full_pool")
    parser.add_argument("--games", type=int, default=5000)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--seed-stride", type=int, default=100000)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--hidden-dims")
    parser.add_argument("--min-avg", type=float)
    parser.add_argument("--min-worst", type=float)
    parser.add_argument("--max-row-std", type=float)
    parser.add_argument("--fail-fast-worst", type=float)
    parser.add_argument("--max-invalid", type=float)
    parser.add_argument("--max-truncated", type=float)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _default_out_dir(config: dict[str, Any], pool: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_id = str(config.get("id") or "seed_config")
    return ROOT / "runs" / f"{config_id}_{pool}_eval_{stamp}"


def main() -> int:
    args = parse_args()
    config = _load_config(args.config)
    baseline = config.get("baseline", {})
    opponents_config = config.get("opponents", {})
    gate = config.get("delivery_gate", {})
    archive_dir = config.get("paths", {}).get("archive_dir")

    hidden_dims = args.hidden_dims or str(baseline.get("hidden_dims", "2048,2048,1024,512"))
    candidates = list(args.candidate)
    if not candidates:
        hinted = _candidate_from_hint(ROOT, baseline.get("checkpoint_hint"))
        candidates = [hinted or str(baseline.get("checkpoint_name"))]
    resolved_candidates = [
        _resolve_checkpoint(ROOT, candidate, archive_dir) for candidate in candidates
    ]

    opponent_names = opponents_config.get(args.pool) or []
    if not opponent_names:
        raise SystemExit(f"No opponents configured for pool: {args.pool}")
    resolved_opponents = [
        _resolve_checkpoint(ROOT, opponent, archive_dir) for opponent in opponent_names
    ]

    out_dir = args.out_dir or _default_out_dir(config, args.pool)
    min_avg = gate.get("min_avg", 0.0) if args.min_avg is None else args.min_avg
    min_worst = gate.get("min_worst", -200.0) if args.min_worst is None else args.min_worst
    max_invalid = gate.get("max_invalid", 0.001) if args.max_invalid is None else args.max_invalid
    max_truncated = (
        gate.get("max_truncated", 0.005)
        if args.max_truncated is None
        else args.max_truncated
    )
    fail_fast_worst = (
        gate.get("fail_fast_worst", -900.0)
        if args.fail_fast_worst is None
        else args.fail_fast_worst
    )
    max_row_std = gate.get("max_row_std") if args.max_row_std is None else args.max_row_std

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "evaluate_seed_repeats.py"),
        "--out-dir",
        str(out_dir),
        "--games",
        str(args.games),
        "--repeats",
        str(args.repeats),
        "--seed",
        str(args.seed),
        "--seed-stride",
        str(args.seed_stride),
        "--hidden-dims",
        hidden_dims,
        "--expected-count",
        str(len(resolved_opponents)),
        "--min-avg",
        str(min_avg),
        "--min-worst",
        str(min_worst),
        "--fail-fast-worst",
        str(fail_fast_worst),
        "--max-invalid",
        str(max_invalid),
        "--max-truncated",
        str(max_truncated),
        "--no-baseline-improvement-required",
    ]
    if max_row_std is not None:
        cmd.extend(["--max-row-std", str(max_row_std)])
    for candidate in resolved_candidates:
        cmd.extend(["--candidate", candidate])
    for opponent in resolved_opponents:
        cmd.extend(["--opponent", opponent])

    print("+ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    if args.dry_run:
        return 0
    return subprocess.run(cmd, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
