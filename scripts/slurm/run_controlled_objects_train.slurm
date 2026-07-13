#!/usr/bin/env bash
#SBATCH --job-name=controlled_hwm
#SBATCH --partition=a40,rtxpro6k,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source scripts/env.sh

RUN_NAME="${RUN_NAME:?RUN_NAME is required}"
SEED="${SEED:?SEED is required}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
MODEL_CONFIG="${MODEL_CONFIG:-cls_hwm}"
OBJECTIVE_CONFIG="${OBJECTIVE_CONFIG:-ema_vicreg}"
HIERARCHY_DEPTH="${HIERARCHY_DEPTH:-1}"
HIERARCHY_STRIDE="${HIERARCHY_STRIDE:-4}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-1}"
ROLLOUT_ALL_LEVELS="${ROLLOUT_ALL_LEVELS:-false}"
ROLLOUT_LAMBDA="${ROLLOUT_LAMBDA:-1.0}"
TRAIN_FROM_LEVEL="${TRAIN_FROM_LEVEL:-0}"

INIT_ARGS=()
if [[ -n "${INIT_CHECKPOINT:-}" ]]; then
  INIT_ARGS+=("training.init_checkpoint=${INIT_CHECKPOINT}")
fi
MODEL_ARGS=()
if [[ -n "${TOKEN_DIM:-}" ]]; then
  MODEL_ARGS+=("model.token_dim=${TOKEN_DIM}")
fi
if [[ -n "${LATENT_DIM:-}" ]]; then
  MODEL_ARGS+=("model.latent_dim=${LATENT_DIM}")
fi
if [[ -n "${LDAD_HORIZON:-}" ]]; then
  MODEL_ARGS+=("model.ldad_horizon=${LDAD_HORIZON}")
fi
OBJECTIVE_ARGS=()
if [[ -n "${LDAD_WEIGHT:-}" ]]; then
  OBJECTIVE_ARGS+=("objective.ldad_weight=${LDAD_WEIGHT}")
fi

python -m puzzle_jepa.train.controlled_objects \
  "model=${MODEL_CONFIG}" \
  "objective=${OBJECTIVE_CONFIG}" \
  "seed=${SEED}" \
  "run_name=${RUN_NAME}" \
  "model.hierarchy_depth=${HIERARCHY_DEPTH}" \
  "model.hierarchy_stride=${HIERARCHY_STRIDE}" \
  "model.rollout_steps=${ROLLOUT_STEPS}" \
  "model.rollout_all_levels=${ROLLOUT_ALL_LEVELS}" \
  "model.rollout_lambda=${ROLLOUT_LAMBDA}" \
  "training.train_from_level=${TRAIN_FROM_LEVEL}" \
  "training.max_steps=${MAX_STEPS:-20000}" \
  "training.eval_every_steps=${EVAL_EVERY_STEPS:-5000}" \
  "training.batch_size=${BATCH_SIZE:-64}" \
  "eval.planning_episodes=${PLANNING_EPISODES:-8}" \
  "eval.planning_candidates=${PLANNING_CANDIDATES:-32}" \
  "output_dir=${OUTPUT_ROOT}/${RUN_NAME}" \
  "${MODEL_ARGS[@]}" \
  "${OBJECTIVE_ARGS[@]}" \
  "${INIT_ARGS[@]}"
