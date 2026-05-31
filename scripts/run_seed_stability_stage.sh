#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/testo-seed-rl}"
RUN_NAME="${RUN_NAME:-seed_stability_200k_from500k_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${ROOT_DIR}/runs/${RUN_NAME}}"
BASE_MODEL="${BASE_MODEL:-${ROOT_DIR}/runs/seed_1_5m_allin_gate_20260530_1645/checkpoints/seed_rl_hands_500016}"
NUM_HANDS="${NUM_HANDS:-200000}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-50000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100000}"
NUM_EVAL_GAMES="${NUM_EVAL_GAMES:-1000}"
CURRICULUM_START_PHASE="${CURRICULUM_START_PHASE:-1}"
CURRICULUM_MAX_PHASE="${CURRICULUM_MAX_PHASE:-3}"
OPPONENT_SCORE_FILE="${OPPONENT_SCORE_FILE:-}"
HARD_OPPONENT_TEMPERATURE="${HARD_OPPONENT_TEMPERATURE:-200}"
HARD_OPPONENT_NEGATIVE_RATIO="${HARD_OPPONENT_NEGATIVE_RATIO:-0}"
OPPONENT_UPDATE_INTERVAL="${OPPONENT_UPDATE_INTERVAL:-10000}"

mkdir -p "${RUN_DIR}/checkpoints"
mkdir -p "${ROOT_DIR}/runs"
echo "${RUN_DIR}" > "${ROOT_DIR}/runs/latest_run_dir"

if [[ -f "${ROOT_DIR}/docs/agent_training_log.md" ]]; then
  cp "${ROOT_DIR}/docs/agent_training_log.md" "${RUN_DIR}/agent_training_log.md"
fi

cd "${ROOT_DIR}"

{
  echo "RUN_DIR=${RUN_DIR}"
  echo "BASE_MODEL=${BASE_MODEL}"
  echo "NUM_HANDS=${NUM_HANDS}"
  echo "LEARNING_RATE=${LEARNING_RATE}"
  echo "CURRICULUM_START_PHASE=${CURRICULUM_START_PHASE}"
  echo "CURRICULUM_MAX_PHASE=${CURRICULUM_MAX_PHASE}"
  echo "OPPONENT_SCORE_FILE=${OPPONENT_SCORE_FILE}"
  echo "HARD_OPPONENT_TEMPERATURE=${HARD_OPPONENT_TEMPERATURE}"
  echo "HARD_OPPONENT_NEGATIVE_RATIO=${HARD_OPPONENT_NEGATIVE_RATIO}"
  echo "OPPONENT_UPDATE_INTERVAL=${OPPONENT_UPDATE_INTERVAL}"
  echo "START_TS=$(date -Is)"
} > "${RUN_DIR}/stage.env"

score_args=()
if [[ -n "${OPPONENT_SCORE_FILE}" ]]; then
  score_args+=(--opponent-score-file "${OPPONENT_SCORE_FILE}")
  score_args+=(--hard-opponent-temperature "${HARD_OPPONENT_TEMPERATURE}")
  score_args+=(--hard-opponent-negative-ratio "${HARD_OPPONENT_NEGATIVE_RATIO}")
fi

/root/miniconda3/bin/python train.py \
  --mode train \
  --hardware-profile rtx4090d \
  --arch seed-rl \
  --model-path "${BASE_MODEL}" \
  --num-hands "${NUM_HANDS}" \
  --learning-rate "${LEARNING_RATE}" \
  --checkpoint-dir "${RUN_DIR}/checkpoints" \
  --metrics-output "${RUN_DIR}/metrics.jsonl" \
  --eval-output "${RUN_DIR}/eval_results.json" \
  --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
  --eval-interval "${EVAL_INTERVAL}" \
  --num-eval-games "${NUM_EVAL_GAMES}" \
  --opponent-update-interval "${OPPONENT_UPDATE_INTERVAL}" \
  --curriculum-start-phase "${CURRICULUM_START_PHASE}" \
  --curriculum-max-phase "${CURRICULUM_MAX_PHASE}" \
  "${score_args[@]}" \
  > "${RUN_DIR}/train.log" 2>&1

code=$?

/root/miniconda3/bin/python scripts/seed_run_report.py \
  "${RUN_DIR}/metrics.jsonl" \
  --title "${RUN_NAME}" \
  --out "${RUN_DIR}/run_report.md" || true

{
  echo
  echo "## Run ${RUN_NAME}"
  echo
  echo "- Finished: $(date -Is)"
  echo "- Exit code: ${code}"
  echo "- Run dir: ${RUN_DIR}"
  echo
  if [[ -f "${RUN_DIR}/run_report.md" ]]; then
    cat "${RUN_DIR}/run_report.md"
  fi
} >> "${RUN_DIR}/agent_training_log.md"

exit "${code}"
