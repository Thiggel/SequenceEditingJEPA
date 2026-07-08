#!/usr/bin/env bash
#SBATCH --job-name=object_dyn_train
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

DATA_CONFIG="${DATA_CONFIG:-semantic_mix}"
MODEL_CONFIG="${MODEL_CONFIG:-cls64_r1}"
OBJECTIVE_CONFIG="${OBJECTIVE_CONFIG:-base}"
RUN_SUFFIX="${RUN_SUFFIX:-manual}"
OUTPUT_ROOT="${PUZZLE_JEPA_WORK_ROOT:-${WORK:?WORK must be set}/sequence-editing-object-dynamics-20260708}"
RUN_ROOT="${OUTPUT_ROOT}/runs/object_dynamics/${DATA_CONFIG}_${MODEL_CONFIG}_${OBJECTIVE_CONFIG}_${RUN_SUFFIX}"

USER_OVERRIDES=()
if [[ -n "${EXTRA_HYDRA_OVERRIDES:-}" ]]; then
  read -r -a USER_OVERRIDES <<< "${EXTRA_HYDRA_OVERRIDES}"
fi

python -m puzzle_jepa.train.object_dynamics \
  --config-name train \
  "data=${DATA_CONFIG}" \
  "model=${MODEL_CONFIG}" \
  "objective=${OBJECTIVE_CONFIG}" \
  "seed=${SEED:-1707}" \
  "output_dir=${RUN_ROOT}" \
  "training.max_steps=${MAX_STEPS:-2000}" \
  "training.batch_size=${BATCH_SIZE:-64}" \
  "training.learning_rate=${LEARNING_RATE:-3.0e-4}" \
  "training.eval_every_steps=${EVAL_EVERY_STEPS:-250}" \
  "eval.probe_train_samples=${PROBE_TRAIN_SAMPLES:-512}" \
  "eval.probe_eval_samples=${PROBE_EVAL_SAMPLES:-256}" \
  "${USER_OVERRIDES[@]}"
