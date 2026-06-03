#!/usr/bin/env python3
"""Offline policy distillation for SEED RL checkpoints."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _checkpoint_base(path: str) -> str:
    path = str(path)
    if path.endswith(".actor.weights.h5"):
        return path[: -len(".actor.weights.h5")]
    if path.endswith(".critic.weights.h5"):
        return path[: -len(".critic.weights.h5")]
    return path


def _load_probs_npz(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    data = np.load(path, allow_pickle=False)
    if "states" not in data:
        raise SystemExit(f"{path} missing states")
    prob_key = None
    for key in ("target_probs", "probs", "action_probs"):
        if key in data:
            prob_key = key
            break
    if prob_key is None:
        raise SystemExit(f"{path} missing target_probs/probs/action_probs")
    states = np.asarray(data["states"], dtype=np.float32)
    probs = np.asarray(data[prob_key], dtype=np.float32)
    metadata = {}
    if "metadata" in data:
        try:
            metadata = json.loads(str(np.asarray(data["metadata"]).item()))
        except Exception:
            metadata = {}
    probs = np.clip(probs, 1e-6, 1.0)
    probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-8)
    return states, probs, metadata


def _row_indices(data: np.lib.npyio.NpzFile, n: int) -> tuple[np.ndarray, list[str]]:
    if "row_ids" not in data:
        return np.zeros(n, dtype=np.int32), ["all"]
    row_ids = np.asarray(data["row_ids"], dtype=np.int32)
    if len(row_ids) != n:
        raise SystemExit(f"row_ids length {len(row_ids)} does not match states {n}")
    if "row_names" in data:
        row_names = [str(x) for x in np.asarray(data["row_names"]).tolist()]
    else:
        row_names = [str(i) for i in sorted(set(row_ids.tolist()))]
    return row_ids, row_names


def _split_by_row(row_ids: np.ndarray, validation_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for row in sorted(set(row_ids.tolist())):
        idx = np.flatnonzero(row_ids == row)
        rng.shuffle(idx)
        val_n = int(round(len(idx) * validation_fraction))
        val_n = min(max(val_n, 1 if len(idx) > 1 else 0), max(len(idx) - 1, 0))
        val_parts.append(idx[:val_n])
        train_parts.append(idx[val_n:])
    train_idx = np.concatenate(train_parts) if train_parts else np.arange(len(row_ids))
    val_idx = np.concatenate(val_parts) if val_parts else np.array([], dtype=np.int64)
    return train_idx.astype(np.int64), val_idx.astype(np.int64)


def _balanced_sample(
    by_row: dict[int, np.ndarray],
    batch_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    rows = np.asarray(sorted(by_row), dtype=np.int32)
    chosen_rows = rng.choice(rows, size=batch_size, replace=True)
    out = np.empty(batch_size, dtype=np.int64)
    for row in rows:
        mask = chosen_rows == row
        count = int(mask.sum())
        if count:
            out[mask] = rng.choice(by_row[int(row)], size=count, replace=True)
    return out


def _mean_kl(model, states: np.ndarray, target_probs: np.ndarray, batch_size: int) -> float:
    if len(states) == 0:
        return 0.0
    vals: list[float] = []
    weights: list[int] = []
    for start in range(0, len(states), batch_size):
        end = min(start + batch_size, len(states))
        pred = model.actor(tf.convert_to_tensor(states[start:end], tf.float32), training=False).numpy()
        pred = np.clip(pred, 1e-6, 1.0)
        pred = pred / (pred.sum(axis=1, keepdims=True) + 1e-8)
        tgt = np.clip(target_probs[start:end], 1e-6, 1.0)
        kl = np.sum(tgt * (np.log(tgt) - np.log(pred)), axis=1)
        vals.append(float(np.mean(kl)))
        weights.append(end - start)
    return float(np.average(vals, weights=weights))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", required=True, help="Initial student checkpoint base path")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--reference-file", type=Path)
    parser.add_argument("--hidden-dims", default="2048,2048,1024,512")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--reference-coef", type=float, default=0.1)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--max-reference-kl", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=20260603)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from poker_ai.config.ppo_config import PPOConfig
    from poker_ai.model.large_ppo_model import LargePPOModel

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "distill_metrics.jsonl"

    raw = np.load(args.dataset, allow_pickle=False)
    states, target_probs, metadata = _load_probs_npz(args.dataset)
    row_ids, row_names = _row_indices(raw, len(states))
    if states.ndim != 2 or target_probs.ndim != 2 or len(states) != len(target_probs):
        raise SystemExit(f"Invalid dataset shapes: states={states.shape}, probs={target_probs.shape}")

    ref_states = np.zeros((0, states.shape[1]), dtype=np.float32)
    ref_probs = np.zeros((0, target_probs.shape[1]), dtype=np.float32)
    if args.reference_file:
        ref_states, ref_probs, _ = _load_probs_npz(args.reference_file)

    hidden_dims = [int(part) for part in args.hidden_dims.split(",") if part.strip()]
    cfg = PPOConfig(
        hidden_dims=hidden_dims,
        lr=float(args.learning_rate),
        batch_size=int(args.batch_size),
        use_lr_warmup=False,
        use_cosine_decay=False,
    )
    model = LargePPOModel(hidden_dims=hidden_dims, config=cfg, lr=float(args.learning_rate), batch_size=int(args.batch_size))
    model.load_weights(_checkpoint_base(args.student))

    opt = tf.keras.optimizers.Adam(learning_rate=float(args.learning_rate))
    rng = np.random.default_rng(int(args.seed))
    train_idx, val_idx = _split_by_row(row_ids, float(args.validation_fraction), int(args.seed))
    by_row = {int(row): train_idx[row_ids[train_idx] == row] for row in sorted(set(row_ids[train_idx].tolist()))}
    by_row = {row: idx for row, idx in by_row.items() if len(idx)}
    if not by_row:
        raise SystemExit("No training rows after split")

    ref_by_row = np.arange(len(ref_states), dtype=np.int64)

    @tf.function(reduce_retracing=True)
    def train_step(batch_states, batch_targets, batch_ref_states, batch_ref_targets, ref_weight):
        with tf.GradientTape() as tape:
            pred = tf.cast(model.actor(batch_states, training=True), tf.float32)
            pred = tf.clip_by_value(pred, 1e-6, 1.0)
            pred = pred / (tf.reduce_sum(pred, axis=1, keepdims=True) + 1e-8)
            target = tf.clip_by_value(tf.cast(batch_targets, tf.float32), 1e-6, 1.0)
            target = target / (tf.reduce_sum(target, axis=1, keepdims=True) + 1e-8)
            teacher_kl = tf.reduce_mean(tf.reduce_sum(target * (tf.math.log(target) - tf.math.log(pred)), axis=1))
            entropy = tf.reduce_mean(-tf.reduce_sum(pred * tf.math.log(pred), axis=1))
            ref_kl = tf.constant(0.0, dtype=tf.float32)
            if tf.shape(batch_ref_states)[0] > 0:
                ref_pred = tf.cast(model.actor(batch_ref_states, training=True), tf.float32)
                ref_pred = tf.clip_by_value(ref_pred, 1e-6, 1.0)
                ref_pred = ref_pred / (tf.reduce_sum(ref_pred, axis=1, keepdims=True) + 1e-8)
                ref_target = tf.clip_by_value(tf.cast(batch_ref_targets, tf.float32), 1e-6, 1.0)
                ref_target = ref_target / (tf.reduce_sum(ref_target, axis=1, keepdims=True) + 1e-8)
                ref_kl = tf.reduce_mean(tf.reduce_sum(ref_target * (tf.math.log(ref_target) - tf.math.log(ref_pred)), axis=1))
            loss = teacher_kl + ref_weight * ref_kl - tf.cast(args.entropy_coef, tf.float32) * entropy
        grads = tape.gradient(loss, model.actor.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, model.max_grad_norm)
        opt.apply_gradients([(g, v) for g, v in zip(grads, model.actor.trainable_variables) if g is not None])
        return loss, teacher_kl, ref_kl, entropy

    run_meta = {
        "student": _checkpoint_base(args.student),
        "dataset": str(args.dataset),
        "reference_file": str(args.reference_file) if args.reference_file else None,
        "rows": row_names,
        "dataset_metadata": metadata,
        "args": vars(args) | {"dataset": str(args.dataset), "out_dir": str(args.out_dir), "reference_file": str(args.reference_file) if args.reference_file else None},
    }
    (args.out_dir / "stage.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    with metrics_path.open("w", encoding="utf-8") as metrics:
        for epoch in range(1, int(args.epochs) + 1):
            losses: list[float] = []
            teacher_losses: list[float] = []
            ref_losses: list[float] = []
            entropies: list[float] = []
            for _ in range(int(args.steps_per_epoch)):
                idx = _balanced_sample(by_row, int(args.batch_size), rng)
                if len(ref_by_row):
                    ref_idx = rng.choice(ref_by_row, size=min(int(args.batch_size), len(ref_by_row)), replace=True)
                    batch_ref_states = ref_states[ref_idx]
                    batch_ref_probs = ref_probs[ref_idx]
                else:
                    batch_ref_states = np.zeros((0, states.shape[1]), dtype=np.float32)
                    batch_ref_probs = np.zeros((0, target_probs.shape[1]), dtype=np.float32)
                loss, teacher_kl, ref_kl, entropy = train_step(
                    tf.convert_to_tensor(states[idx], tf.float32),
                    tf.convert_to_tensor(target_probs[idx], tf.float32),
                    tf.convert_to_tensor(batch_ref_states, tf.float32),
                    tf.convert_to_tensor(batch_ref_probs, tf.float32),
                    tf.convert_to_tensor(float(args.reference_coef if len(ref_by_row) else 0.0), tf.float32),
                )
                losses.append(float(loss.numpy()))
                teacher_losses.append(float(teacher_kl.numpy()))
                ref_losses.append(float(ref_kl.numpy()))
                entropies.append(float(entropy.numpy()))

            val_teacher_kl = _mean_kl(model, states[val_idx], target_probs[val_idx], int(args.batch_size)) if len(val_idx) else 0.0
            ref_eval_kl = _mean_kl(model, ref_states, ref_probs, int(args.batch_size)) if len(ref_states) else 0.0
            ckpt = args.out_dir / "checkpoints" / f"seed_distilled_epoch{epoch}"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            model.save_weights(str(ckpt))
            row_metrics: dict[str, float] = {}
            for row in sorted(set(row_ids[val_idx].tolist())) if len(val_idx) else []:
                ridx = val_idx[row_ids[val_idx] == row]
                name = row_names[row] if row < len(row_names) else str(row)
                row_metrics[name] = _mean_kl(model, states[ridx], target_probs[ridx], int(args.batch_size))
            record = {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "teacher_kl": float(np.mean(teacher_losses)),
                "reference_kl_train": float(np.mean(ref_losses)),
                "entropy": float(np.mean(entropies)),
                "val_teacher_kl": val_teacher_kl,
                "reference_kl": ref_eval_kl,
                "checkpoint": str(ckpt),
                "row_val_teacher_kl": row_metrics,
                "status": "FAIL" if ref_eval_kl > float(args.max_reference_kl) else "OK",
            }
            metrics.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            metrics.flush()
            print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)
            if ref_eval_kl > float(args.max_reference_kl):
                logging.warning("Stopping: reference KL %.6f > %.6f", ref_eval_kl, float(args.max_reference_kl))
                break

    final_path = args.out_dir / "seed_distilled_final"
    model.save_weights(str(final_path))
    print(json.dumps({"final": str(final_path), "metrics": str(metrics_path)}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
