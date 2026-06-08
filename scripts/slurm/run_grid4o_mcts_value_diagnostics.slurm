#!/usr/bin/env bash
#SBATCH --job-name=puzzle_grid4o_mcts
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/sudoku_jepa_5m_goal_energy_hwm_l1"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    SCORE="goal_energy"
    DEPTH=8
    ;;
  1)
    SCORE="goal_energy"
    DEPTH=16
    ;;
  2)
    SCORE="latent_goal"
    DEPTH=8
    ;;
  3)
    SCORE="latent_goal"
    DEPTH=16
    ;;
  *)
    echo "Unknown SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

OUT_DIR="${RUN_ROOT}/diagnostics_mcts_${SCORE}_d${DEPTH}"

python -m puzzle_jepa.eval.diagnostics \
  --run-root "$RUN_ROOT" \
  --output-dir "$OUT_DIR" \
  --rank-examples 0 \
  --drift-examples 0 \
  --trace-examples 0 \
  --planning-examples 0 \
  --cem-examples 0 \
  --mcts-examples 128 \
  --mcts-score "$SCORE" \
  --mcts-simulations 512 \
  --mcts-depth "$DEPTH" \
  --mcts-exploration 1.0 \
  --mcts-expansion-actions 64 \
  --mcts-debug-examples 8 \
  --mcts-debug-actions 16 \
  --max-unroll-steps 81 \
  --seed $((4100 + SLURM_ARRAY_TASK_ID))
