#!/usr/bin/env bash
#SBATCH --job-name=puzzle_diag2a
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
  "${PUZZLE_JEPA_WORK_ROOT}/runs/sudoku_jepa_5m_mix50_50_rollout_n2"
  "${PUZZLE_JEPA_WORK_ROOT}/runs/sudoku_jepa_5m_mix50_50_rollout_n4"
  "${PUZZLE_JEPA_WORK_ROOT}/runs/maze_jepa_5m_oracle_rollout_n2"
  "${PUZZLE_JEPA_WORK_ROOT}/runs/maze_jepa_5m_oracle_rollout_n4"
)

RANK_EXAMPLES=(256 256 64 64)
DRIFT_EXAMPLES=(32 32 8 8)
TRACE_EXAMPLES=(8 8 4 4)
PLANNING_EXAMPLES=(8 8 2 2)
MAX_UNROLL_STEPS=(81 81 512 512)

INDEX="${SLURM_ARRAY_TASK_ID}"
python -m puzzle_jepa.eval.diagnostics \
  --run-root "${RUN_ROOTS[$INDEX]}" \
  --rank-examples "${RANK_EXAMPLES[$INDEX]}" \
  --drift-examples "${DRIFT_EXAMPLES[$INDEX]}" \
  --trace-examples "${TRACE_EXAMPLES[$INDEX]}" \
  --planning-examples "${PLANNING_EXAMPLES[$INDEX]}" \
  --max-unroll-steps "${MAX_UNROLL_STEPS[$INDEX]}" \
  --horizons 1 2 4 10 20 \
  --depth-fractions 0.0 0.25 0.5 0.75 \
  --planning-beam-size 4 \
  --planning-branch-size 8
