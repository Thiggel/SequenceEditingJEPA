#!/usr/bin/env bash
#SBATCH --job-name=puzzle_diag3a
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-3

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

RUN_ROOTS=(
  "${PUZZLE_JEPA_WORK_ROOT}/runs/sudoku_jepa_5m_local_direct_uniform"
  "${PUZZLE_JEPA_WORK_ROOT}/runs/sudoku_jepa_5m_local_direct_weighted"
  "${PUZZLE_JEPA_WORK_ROOT}/runs/sudoku_jepa_5m_local_residual_weighted"
  "${PUZZLE_JEPA_WORK_ROOT}/runs/sudoku_jepa_5m_local_direct_changed_only"
)

RUN_ROOT="${RUN_ROOTS[$SLURM_ARRAY_TASK_ID]}"
python -m puzzle_jepa.eval.diagnostics \
  --run-root "$RUN_ROOT" \
  --rank-examples 256 \
  --drift-examples 32 \
  --trace-examples 4 \
  --planning-examples 8 \
  --max-unroll-steps 64 \
  --horizons 1 2 4 10 20 \
  --depth-fractions 0.0 0.25 0.5 0.75 \
  --planning-beam-size 4 \
  --planning-branch-size 8
