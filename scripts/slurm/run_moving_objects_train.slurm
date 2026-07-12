#!/usr/bin/env bash
#SBATCH --job-name=motion_jepa
#SBATCH --partition=a40,rtxpro6k,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source scripts/env.sh

LATENT_DIM="${LATENT_DIM:?LATENT_DIM is required}"
MAX_OBJECTS="${MAX_OBJECTS:?MAX_OBJECTS is required}"
MIN_OBJECTS="${MIN_OBJECTS:-1}"
SEED="${SEED:?SEED is required}"
RUN_SUFFIX="${RUN_SUFFIX:-bottleneck_grid}"
OBJECTIVE_CONFIG="${OBJECTIVE_CONFIG:-ema_vicreg}"
DATA_CONFIG="${DATA_CONFIG:-reflected_motion}"
RUN_NAME="motion_n${MAX_OBJECTS}_z${LATENT_DIM}_${RUN_SUFFIX}_seed${SEED}"

python -m puzzle_jepa.train.moving_objects \
  "data=${DATA_CONFIG}" \
  "objective=${OBJECTIVE_CONFIG}" \
  "seed=${SEED}" \
  "model.latent_dim=${LATENT_DIM}" \
  "data.min_objects=${MIN_OBJECTS}" \
  "data.max_objects=${MAX_OBJECTS}" \
  "training.max_steps=${MAX_STEPS:-5000}" \
  "training.learning_rate=${LEARNING_RATE:-3.0e-4}" \
  "training.eval_every_steps=${MAX_STEPS:-5000}" \
  "output_dir=${PUZZLE_JEPA_WORK_ROOT}/runs/moving_objects/${RUN_NAME}"
