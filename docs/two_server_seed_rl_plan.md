# Two-Server SEED RL Plan

This plan keeps training and acceptance evaluation separate now that two GPU
servers are available.

## Roles

- 2-GPU server: candidate generation and fast prescreening.
  - Run short smoke training or evaluation jobs.
  - Use independent `CUDA_VISIBLE_DEVICES=0/1` processes for parallel matrix
    prescreens.
  - Keep long jobs behind repeat-gate checkpoints; do not promote locally.
- 1-GPU server: independent acceptance gate.
  - Run final `seed_safety_validate.py` or `evaluate_seed_repeats.py`
    confirmations.
  - Treat this machine as the unbiased verifier for any checkpoint produced on
    the 2-GPU server.

## Current Technical Direction

- The model is not delivery-ready.
- Do not continue weighted PPO or blind reference-PPO repair from rejected
  checkpoints.
- Robust repeated evaluation is mandatory.  Mean-only teacher selection caused
  false positives, especially on high-variance rows such as `900236`.
- The immediate next engineering step is infrastructure for safer candidate
  generation and faster repeat prescreening.  Opponent-profile state expansion
  remains a planned branch, but should not be mixed into the same smoke that
  validates the two-server pipeline.

## 2-GPU Usage

Use two independent processes before attempting TensorFlow `MirroredStrategy`.
SEED RL currently spends meaningful time in environment simulation and batched
inference, so simple process-level parallelism is lower risk and easier to
inspect.

Example prescreen:

```bash
python scripts/evaluate_seed_repeats_parallel.py \
  --candidate /root/testo-seed-rl/runs/seed_selfplay_3m_from600k_20260531_093252/checkpoints/seed_rl_hands_3600064 \
  --opponent /path/to/opponent_a \
  --opponent /path/to/opponent_b \
  --cuda-device 0 \
  --cuda-device 1 \
  --games 500 \
  --repeats 3 \
  --out-dir /root/testo-seed-rl/runs/prescreen_3600064
```

The script writes shard JSONL files, merged repeat JSONL files, an aggregate
matrix, `repeat_report.md`, and `gate_report.md`.

## Promotion Rule

A checkpoint can only become a new research baseline after:

1. It passes a 2-GPU prescreen on the candidate server.
2. It passes an independent repeat gate on the 1-GPU server.
3. The result is logged in `docs/agent_training_log.md`.

## Do Not Do Yet

- Do not run two unrelated long training jobs just because two GPUs exist.
- Do not treat 20/100-game smoke results as model quality signals.
- Do not promote a checkpoint from the 2-GPU server without 1-GPU confirmation.
- Do not store server passwords or API keys in this repository.
