#!/usr/bin/env bash
#SBATCH --job-name=puzzle_diag4a_reset_ge
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-2

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

RUNS=(
  sudoku_jepa_5m_goal_energy_hwm_l1
  sudoku_jepa_5m_goal_energy_hwm_l2_span9
  sudoku_jepa_5m_goal_energy_hwm_l3_span3
)

RUN_NAME="${RUNS[$SLURM_ARRAY_TASK_ID]}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/${RUN_NAME}"

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
