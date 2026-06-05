#!/usr/bin/env bash
#SBATCH --job-name=puzzle_grid4k_ltr
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

LABELS=(
  remaining_discounted
  latent_goal_distance
)
SUFFIXES=(
  remaining_g099
  latent_goal
)
SEEDS=(
  831
  832
)

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
LABEL="${LABELS[$INDEX]}"
SUFFIX="${SUFFIXES[$INDEX]}"
SEED="${SEEDS[$INDEX]}"
RUN_NAME="sudoku_jepa_5m_value_method_l1_ltr_listnet_${SUFFIX}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/${RUN_NAME}"

python -m puzzle_jepa.train.grid0 \
  --config-name grid4a_sudoku_jepa_5m_goal_energy_cem_l1 \
  "seed=${SEED}" \
  "output_dir=${RUN_ROOT}" \
  "model.hierarchy_levels=1" \
  "training.hierarchy_weight=0.0" \
  "training.goal_energy_weight=0.0" \
  "+training.goal_ranking_weight=1.0" \
  "+training.goal_ranking_label=${LABEL}" \
  "+training.goal_ranking_batch_size=16" \
  "+training.goal_ranking_candidates=128" \
  "+training.goal_ranking_discount=0.99" \
  "+training.goal_ranking_label_temperature=1.0" \
  "+training.goal_ranking_score_temperature=0.05" \
  "+training.goal_energy_aux_sampling=stratified_puzzle" \
  "+training.goal_energy_aux_puzzles=16" \
  "+training.goal_energy_aux_states_per_puzzle=1"

python -m puzzle_jepa.eval.diagnostics \
  --run-root "$RUN_ROOT" \
  --output-dir "$RUN_ROOT/diagnostics_reset_ltr_listnet" \
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
