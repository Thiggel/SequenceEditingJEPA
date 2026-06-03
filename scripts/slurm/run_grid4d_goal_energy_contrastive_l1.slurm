#!/usr/bin/env bash
#SBATCH --job-name=puzzle_grid4d_ge_contrast
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-5

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

LOSSES=(nce infonce margin nce infonce margin)
MONOTONE_WEIGHTS=(0.0 0.0 0.0 1.0 1.0 1.0)
SUFFIXES=(
  nce
  infonce
  margin
  nce_mono
  infonce_mono
  margin_mono
)

LOSS_NAME="${LOSSES[$SLURM_ARRAY_TASK_ID]}"
MONOTONE_WEIGHT="${MONOTONE_WEIGHTS[$SLURM_ARRAY_TASK_ID]}"
SUFFIX="${SUFFIXES[$SLURM_ARRAY_TASK_ID]}"
RUN_NAME="sudoku_jepa_5m_goal_energy_contrast_l1_${SUFFIX}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/${RUN_NAME}"
SEED=$((701 + SLURM_ARRAY_TASK_ID))

sleep $((SLURM_ARRAY_TASK_ID * 30))

python -m puzzle_jepa.train.grid0 \
  --config-name grid4a_sudoku_jepa_5m_goal_energy_cem_l1 \
  "seed=${SEED}" \
  "output_dir=${RUN_ROOT}" \
  "+training.goal_energy_contrastive_loss=${LOSS_NAME}" \
  "+training.goal_energy_contrastive_weight=1.0" \
  "+training.goal_energy_aux_batch_size=64" \
  "+training.goal_energy_aux_regression_weight=0.0" \
  "+training.goal_energy_negative_count=8" \
  "+training.goal_energy_contrastive_temperature=0.1" \
  "+training.goal_energy_contrastive_margin=0.05" \
  "+training.goal_energy_monotonicity_weight=${MONOTONE_WEIGHT}" \
  "+training.goal_energy_monotonicity_margin=0.0" \
  "model.hierarchy_levels=1" \
  "training.hierarchy_weight=0.0"

python -m puzzle_jepa.eval.diagnostics \
  --run-root "$RUN_ROOT" \
  --output-dir "$RUN_ROOT/diagnostics_reset_goal_energy" \
  --rank-examples 0 \
  --drift-examples 0 \
  --trace-examples 0 \
  --planning-examples 128 \
  --planning-score goal_energy \
  --planning-beam-size 4 \
  --planning-branch-size 8 \
  --reset-cadences 4 \
  --cem-examples 0 \
  --max-unroll-steps 81

python -m puzzle_jepa.eval.diagnostics \
  --run-root "$RUN_ROOT" \
  --output-dir "$RUN_ROOT/diagnostics_reset_oracle_calibration" \
  --rank-examples 0 \
  --drift-examples 0 \
  --trace-examples 0 \
  --planning-examples 128 \
  --planning-score latent_goal \
  --planning-beam-size 4 \
  --planning-branch-size 8 \
  --reset-cadences 4 \
  --cem-examples 0 \
  --max-unroll-steps 81
