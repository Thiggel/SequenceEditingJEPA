#!/usr/bin/env bash
#SBATCH --job-name=arc_jepa_train
#SBATCH --partition=rtxpro6k,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

VARIANTS=(raw_grid_energy proposal_energy jepa_energy)
if [[ -n "${VARIANT:-}" ]]; then
  SELECTED_VARIANT="${VARIANT}"
else
  INDEX="${SLURM_ARRAY_TASK_ID:-0}"
  if (( INDEX < 0 || INDEX >= ${#VARIANTS[@]} )); then
    echo "Unknown SLURM_ARRAY_TASK_ID=${INDEX}; valid range is 0-$(( ${#VARIANTS[@]} - 1 ))" >&2
    exit 2
  fi
  SELECTED_VARIANT="${VARIANTS[$INDEX]}"
fi

export ARC_AGI_ROOT="${ARC_AGI_ROOT:-${WORK:?WORK must be set}/datasets/arc-agi}"
if [[ ! -d "${ARC_AGI_ROOT}/data/training" ]]; then
  mkdir -p "$(dirname "${ARC_AGI_ROOT}")"
  git clone --depth 1 https://github.com/fchollet/arc-agi.git "${ARC_AGI_ROOT}"
fi

RUN_NAME="arc_${SELECTED_VARIANT}_${RUN_SUFFIX:-firstpass}"
OUTPUT_ROOT="${PUZZLE_JEPA_WORK_ROOT:-${WORK:?WORK must be set}/sequence-editing-arc-20260707}"
RUN_ROOT="${OUTPUT_ROOT}/runs/arc_jepa/${RUN_NAME}"

USER_OVERRIDES=()
if [[ -n "${EXTRA_HYDRA_OVERRIDES:-}" ]]; then
  read -r -a USER_OVERRIDES <<< "${EXTRA_HYDRA_OVERRIDES}"
fi

python -m puzzle_jepa.train.arc_jepa \
  --config-name arc_jepa \
  "variant=${SELECTED_VARIANT}" \
  "seed=${SEED:-1707}" \
  "output_dir=${RUN_ROOT}" \
  "data.data_root=${ARC_AGI_ROOT}" \
  "data.task_limit=${TASK_LIMIT:-120}" \
  "data.eval_task_count=${EVAL_TASK_COUNT:-20}" \
  "training.max_steps=${MAX_STEPS:-1500}" \
  "training.batch_size=${BATCH_SIZE:-16}" \
  "training.eval_every_steps=${EVAL_EVERY_STEPS:-250}" \
  "sampler.max_actions=${MAX_ACTIONS:-700}" \
  "eval.max_actions=${EVAL_MAX_ACTIONS:-700}" \
  "${USER_OVERRIDES[@]}"
