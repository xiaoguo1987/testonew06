#!/usr/bin/env python3
"""Check a SEED RL checkpoint against a fixed reference distillation buffer."""

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


def _checkpoint_base(path: str) -> str:
    if path.endswith(".actor.weights.h5"):
        return path[: -len(".actor.weights.h5")]
    return path


def _load_reference(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    if "states" not in data:
        raise SystemExit(f"Reference file missing states: {path}")
    prob_key = None
    for key in ("target_probs", "probs", "action_probs"):
        if key in data:
            prob_key = key
            break
    if prob_key is None:
        raise SystemExit(f"Reference file missing target_probs/probs: {path}")

    states = np.asarray(data["states"], dtype=np.float32)
    target = np.asarray(data[prob_key], dtype=np.float32)
    if len(states) != len(target):
        raise SystemExit(f"Reference states/probs length mismatch: {states.shape} vs {target.shape}")
    target = np.clip(target, 1e-6, 1.0)
    target = target / (target.sum(axis=1, keepdims=True) + 1e-8)
    return states, target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Checkpoint base path")
    parser.add_argument("--reference-file", required=True, type=Path)
    parser.add_argument("--hidden-dims", default="2048,2048,1024,512")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-mean-kl", type=float, default=None)
    parser.add_argument("--max-p95-kl", type=float, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--fail-exit-code", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from poker_ai.config.ppo_config import PPOConfig
    from poker_ai.model.large_ppo_model import LargePPOModel

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    hidden_dims = [int(part) for part in args.hidden_dims.split(",") if part.strip()]
    cfg = PPOConfig(hidden_dims=hidden_dims)
    model = LargePPOModel(hidden_dims=hidden_dims, config=cfg)
    model_path = _checkpoint_base(args.model)
    model.load_weights(model_path)
    states, target = _load_reference(args.reference_file)
    if states.ndim != 2 or states.shape[1] != model.input_dim:
        raise SystemExit(f"Reference states shape {states.shape} does not match input_dim={model.input_dim}")
    if target.ndim != 2 or target.shape[1] != model.output_dim:
        raise SystemExit(f"Reference probs shape {target.shape} does not match output_dim={model.output_dim}")

    batch_size = max(1, int(args.batch_size))
    all_kls: list[np.ndarray] = []
    for start in range(0, len(states), batch_size):
        end = min(start + batch_size, len(states))
        probs, _ = model.predict_batch(states[start:end])
        probs = np.clip(np.asarray(probs, dtype=np.float32), 1e-6, 1.0)
        probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-8)
        tgt = target[start:end]
        kl = np.sum(tgt * (np.log(tgt) - np.log(probs)), axis=1)
        all_kls.append(kl.astype(np.float64))

    kls = np.concatenate(all_kls) if all_kls else np.array([], dtype=np.float64)
    summary = {
        "model": model_path,
        "reference_file": str(args.reference_file),
        "rows": int(len(kls)),
        "mean_kl": float(np.mean(kls)) if len(kls) else 0.0,
        "p50_kl": float(np.percentile(kls, 50)) if len(kls) else 0.0,
        "p95_kl": float(np.percentile(kls, 95)) if len(kls) else 0.0,
        "max_kl": float(np.max(kls)) if len(kls) else 0.0,
    }
    failures: list[str] = []
    if args.max_mean_kl is not None and summary["mean_kl"] > args.max_mean_kl:
        failures.append(f"mean_kl {summary['mean_kl']:.4f} > {args.max_mean_kl:.4f}")
    if args.max_p95_kl is not None and summary["p95_kl"] > args.max_p95_kl:
        failures.append(f"p95_kl {summary['p95_kl']:.4f} > {args.max_p95_kl:.4f}")
    summary["status"] = "FAIL" if failures else "PASS"
    summary["failures"] = failures

    text = json.dumps(summary, ensure_ascii=False, sort_keys=True)
    print(text, flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return args.fail_exit_code if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
