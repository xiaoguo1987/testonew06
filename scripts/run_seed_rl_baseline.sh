#!/usr/bin/env bash
set -euo pipefail

HANDS="${1:-100000}"
RUN_NAME="${RUN_NAME:-seed_rl_${HANDS}_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-runs/${RUN_NAME}}"

mkdir -p "${RUN_DIR}"

python train.py \
  --mode train \
  --hardware-profile rtx4090d \
  --arch seed-rl \
  --num-hands "${HANDS}" \
  --checkpoint-dir "${RUN_DIR}/checkpoints" \
  --metrics-output "${RUN_DIR}/metrics.jsonl" \
  --eval-output "${RUN_DIR}/eval_results.json" \
  --checkpoint-interval "${CHECKPOINT_INTERVAL:-50000}" \
  --eval-interval "${EVAL_INTERVAL:-50000}" \
  --num-eval-games "${NUM_EVAL_GAMES:-1000}" \
  --eval-seed "${EVAL_SEED:-20260530}" \
  --skip-final-eval
