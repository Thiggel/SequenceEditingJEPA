#!/usr/bin/env bash
#SBATCH --job-name=puzzle_grid4i_valuefix
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/sudoku_jepa_5m_value_method_l1_discounted_reachability_g099"

python -m puzzle_jepa.eval.diagnostics \
  --run-root "$RUN_ROOT" \
  --output-dir "$RUN_ROOT/diagnostics_reset_goal_value_fixed" \
  --rank-examples 0 \
  --drift-examples 0 \
  --trace-examples 0 \
  --planning-examples 128 \
  --planning-score goal_value \
  --planning-beam-size 4 \
  --planning-branch-size 8 \
  --reset-cadences 4 \
  --cem-examples 0 \
  --max-unroll-steps 81
