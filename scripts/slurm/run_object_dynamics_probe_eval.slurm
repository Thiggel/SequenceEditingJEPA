#!/usr/bin/env bash
#SBATCH --job-name=object_dyn_probe
#SBATCH --partition=a40,rtxpro6k,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

if [[ -z "${RUN_NAME:-}" ]]; then
  echo "RUN_NAME must be set." >&2
  exit 2
fi

OUTPUT_ROOT="${PUZZLE_JEPA_WORK_ROOT:-${WORK:?WORK must be set}/sequence-editing-object-dynamics-20260708}"
RUN_DIR="${OUTPUT_ROOT}/runs/object_dynamics/${RUN_NAME}"
CHECKPOINT="${RUN_DIR}/checkpoint.pt"
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi

probe_args=()
if [[ -n "${PROBE_TRAJECTORY_KIND:-}" ]]; then
  probe_args+=(--probe-trajectory-kind "${PROBE_TRAJECTORY_KIND}")
fi

python -m puzzle_jepa.eval.object_dynamics_checkpoint \
  --checkpoint "${CHECKPOINT}" \
  --output "${RUN_DIR}/${PROBE_OUTPUT_NAME:-probe_eval_balanced_v4.json}" \
  --device auto \
  --train-samples "${PROBE_TRAIN_SAMPLES:-512}" \
  --eval-samples "${PROBE_EVAL_SAMPLES:-256}" \
  --batch-size "${PROBE_BATCH_SIZE:-64}" \
  --steps "${PROBE_STEPS:-150}" \
  --learning-rate "${PROBE_LEARNING_RATE:-1.0e-2}" \
  "${probe_args[@]}"
