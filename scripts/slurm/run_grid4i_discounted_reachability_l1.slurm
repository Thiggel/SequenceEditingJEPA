#!/usr/bin/env bash
#SBATCH --job-name=puzzle_grid4i_reach
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

RUN_NAME="sudoku_jepa_5m_value_method_l1_discounted_reachability_g099"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/${RUN_NAME}"

python -m puzzle_jepa.train.grid0 \
  --config-name grid4a_sudoku_jepa_5m_goal_energy_cem_l1 \
  "seed=822" \
  "output_dir=${RUN_ROOT}" \
  "model.hierarchy_levels=1" \
  "training.hierarchy_weight=0.0" \
  "training.goal_energy_weight=0.0" \
  "+training.goal_terminal_correctness_weight=1.0" \
  "+training.goal_terminal_target_mode=discounted_reachability" \
  "+training.goal_terminal_discount=0.99" \
  "+training.goal_terminal_batch_size=256" \
  "+training.goal_terminal_positive_fraction=0.5" \
  "+training.goal_terminal_regression_weight=0.0"

python -m puzzle_jepa.eval.diagnostics \
  --run-root "$RUN_ROOT" \
  --output-dir "$RUN_ROOT/diagnostics_reset_discounted_reachability" \
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
