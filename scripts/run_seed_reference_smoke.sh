#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/testo-seed-rl}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
RUN_NAME="${RUN_NAME:-seed_reference_smoke_10k_from3600k_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${ROOT_DIR}/runs/${RUN_NAME}}"
ANCHOR_NAME="${ANCHOR_NAME:-seed_rl_hands_3600064}"
NUM_HANDS="${NUM_HANDS:-10000}"
REF_STATES_PER_OPPONENT="${REF_STATES_PER_OPPONENT:-5000}"
REF_MAX_HANDS_PER_OPPONENT="${REF_MAX_HANDS_PER_OPPONENT:-30000}"
REF_COEF="${REF_COEF:-0.2}"
REF_BATCH_SIZE="${REF_BATCH_SIZE:-512}"
ANCHOR_KL_COEF="${ANCHOR_KL_COEF:-0.05}"
MAX_REFERENCE_KL="${MAX_REFERENCE_KL:-0.75}"
MAX_REFERENCE_MEAN_KL="${MAX_REFERENCE_MEAN_KL:-0.75}"
MAX_REFERENCE_P95_KL="${MAX_REFERENCE_P95_KL:-2.0}"
MAX_ANCHOR_KL="${MAX_ANCHOR_KL:-0.75}"
LEARNING_RATE="${LEARNING_RATE:-1e-7}"
PPO_CLIP="${PPO_CLIP:-0.04}"
PPO_ENTROPY_COEF="${PPO_ENTROPY_COEF:-0.01}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10000}"
NUM_EVAL_GAMES="${NUM_EVAL_GAMES:-100}"
VALIDATION_GAMES="${VALIDATION_GAMES:-100}"
VALIDATION_REPEATS="${VALIDATION_REPEATS:-3}"
OPPONENT_ENSEMBLE_SIZE="${OPPONENT_ENSEMBLE_SIZE:-3}"
OPPONENT_ROSTER_FILE="${OPPONENT_ROSTER_FILE:-${ROOT_DIR}/configs/seed_roster_3600064_safety.json}"
OPPONENT_NAMES="${OPPONENT_NAMES:-seed_rl_hands_1000262 seed_rl_hands_1500036 seed_rl_hands_600040 seed_rl_hands_1000054 seed_rl_hands_600047 seed_rl_hands_700169}"

cd "${ROOT_DIR}"
mkdir -p "${RUN_DIR}/checkpoints" "${RUN_DIR}/reference" "${ROOT_DIR}/runs"
echo "${RUN_DIR}" > "${ROOT_DIR}/runs/latest_run_dir"

resolve_checkpoint() {
  local name="$1"
  local found
  found="$(
    find "${ROOT_DIR}/runs" "${ROOT_DIR}/checkpoints" /root/autodl-tmp/poker_checkpoints \
      -name "${name}.actor.weights.h5" -print -quit 2>/dev/null || true
  )"
  if [[ -z "${found}" ]]; then
    return 1
  fi
  printf '%s\n' "${found%.actor.weights.h5}"
}

shell_quote() {
  printf '%q' "$1"
}

if [[ -z "${ANCHOR:-}" ]]; then
  ANCHOR="$(resolve_checkpoint "${ANCHOR_NAME}" || true)"
fi
if [[ -z "${ANCHOR}" ]]; then
  echo "Unable to resolve anchor checkpoint ${ANCHOR_NAME}" >&2
  exit 2
fi

opponent_args=()
opponent_paths=()
for name in ${OPPONENT_NAMES}; do
  path="$(resolve_checkpoint "${name}" || true)"
  if [[ -z "${path}" ]]; then
    echo "Unable to resolve opponent checkpoint ${name}" >&2
    exit 2
  fi
  opponent_paths+=("${path}")
  opponent_args+=(--opponent "${path}")
done

REF_FILE="${REF_FILE:-${RUN_DIR}/reference/${ANCHOR_NAME}_hard_subset_reference.npz}"

{
  echo "RUN_DIR=${RUN_DIR}"
  echo "ANCHOR=${ANCHOR}"
  echo "NUM_HANDS=${NUM_HANDS}"
  echo "REF_FILE=${REF_FILE}"
  echo "REF_STATES_PER_OPPONENT=${REF_STATES_PER_OPPONENT}"
  echo "REF_COEF=${REF_COEF}"
  echo "REF_BATCH_SIZE=${REF_BATCH_SIZE}"
  echo "ANCHOR_KL_COEF=${ANCHOR_KL_COEF}"
  echo "MAX_REFERENCE_KL=${MAX_REFERENCE_KL}"
  echo "MAX_REFERENCE_MEAN_KL=${MAX_REFERENCE_MEAN_KL}"
  echo "MAX_REFERENCE_P95_KL=${MAX_REFERENCE_P95_KL}"
  echo "MAX_ANCHOR_KL=${MAX_ANCHOR_KL}"
  echo "LEARNING_RATE=${LEARNING_RATE}"
  echo "PPO_CLIP=${PPO_CLIP}"
  echo "PPO_ENTROPY_COEF=${PPO_ENTROPY_COEF}"
  echo "OPPONENT_ROSTER_FILE=${OPPONENT_ROSTER_FILE}"
  echo "OPPONENTS=${opponent_paths[*]}"
  echo "START_TS=$(date -Is)"
} > "${RUN_DIR}/stage.env"

if [[ -f "${ROOT_DIR}/docs/agent_training_log.md" ]]; then
  cp "${ROOT_DIR}/docs/agent_training_log.md" "${RUN_DIR}/agent_training_log.md"
fi

"${PYTHON}" scripts/build_seed_reference_buffer.py \
  --model "${ANCHOR}" \
  --states-per-opponent "${REF_STATES_PER_OPPONENT}" \
  --max-hands-per-opponent "${REF_MAX_HANDS_PER_OPPONENT}" \
  --out "${REF_FILE}" \
  "${opponent_args[@]}" \
  > "${RUN_DIR}/build_reference.log" 2>&1

"${PYTHON}" scripts/check_seed_reference_kl.py \
  --model "${ANCHOR}" \
  --reference-file "${REF_FILE}" \
  --max-mean-kl 0.02 \
  --max-p95-kl 0.05 \
  --out "${RUN_DIR}/anchor_reference_kl.json" \
  > "${RUN_DIR}/anchor_reference_kl.log" 2>&1

validation_cmd="$(shell_quote "${PYTHON}") scripts/seed_safety_validate.py --checkpoint \"\$SEED_RL_CHECKPOINT\""
validation_cmd+=" --games ${VALIDATION_GAMES} --repeats ${VALIDATION_REPEATS} --expected-count ${#opponent_paths[@]}"
validation_cmd+=" --min-avg -100 --min-worst -350 --fail-fast-worst -900 --max-invalid 0.001 --max-truncated 0.005"
validation_cmd+=" --reference-file $(shell_quote "${REF_FILE}") --max-reference-mean-kl ${MAX_REFERENCE_MEAN_KL} --max-reference-p95-kl ${MAX_REFERENCE_P95_KL}"
for opponent in "${opponent_paths[@]}"; do
  validation_cmd+=" --opponent $(shell_quote "${opponent}")"
done

roster_args=()
if [[ -f "${OPPONENT_ROSTER_FILE}" ]]; then
  roster_args+=(--opponent-roster-file "${OPPONENT_ROSTER_FILE}")
fi

"${PYTHON}" train.py \
  --mode train \
  --hardware-profile rtx4090d \
  --arch seed-rl \
  --model-path "${ANCHOR}" \
  --num-hands "${NUM_HANDS}" \
  --learning-rate "${LEARNING_RATE}" \
  --ppo-clip "${PPO_CLIP}" \
  --ppo-entropy-coef "${PPO_ENTROPY_COEF}" \
  --anchor-policy-path "${ANCHOR}" \
  --anchor-kl-coef "${ANCHOR_KL_COEF}" \
  --max-anchor-kl "${MAX_ANCHOR_KL}" \
  --anchor-kl-patience 3 \
  --reference-distill-file "${REF_FILE}" \
  --reference-distill-coef "${REF_COEF}" \
  --reference-distill-batch-size "${REF_BATCH_SIZE}" \
  --max-reference-kl "${MAX_REFERENCE_KL}" \
  --reference-kl-patience 3 \
  --checkpoint-dir "${RUN_DIR}/checkpoints" \
  --metrics-output "${RUN_DIR}/metrics.jsonl" \
  --eval-output "${RUN_DIR}/eval_results.json" \
  --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
  --eval-interval "${EVAL_INTERVAL}" \
  --num-eval-games "${NUM_EVAL_GAMES}" \
  --opponent-update-interval 10000 \
  --opponent-ensemble-size "${OPPONENT_ENSEMBLE_SIZE}" \
  --curriculum-start-phase 3 \
  --curriculum-max-phase 3 \
  "${roster_args[@]}" \
  --checkpoint-validation-command "${validation_cmd}" \
  --stop-on-validation-failure \
  --validation-start-hands 0 \
  --skip-final-buffer-train \
  > "${RUN_DIR}/train.log" 2>&1

code=$?

"${PYTHON}" scripts/seed_run_report.py \
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
  echo "- Reference file: ${REF_FILE}"
  echo
  if [[ -f "${RUN_DIR}/run_report.md" ]]; then
    cat "${RUN_DIR}/run_report.md"
  fi
} >> "${RUN_DIR}/agent_training_log.md"

exit "${code}"
