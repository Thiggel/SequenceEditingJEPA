#!/usr/bin/env bash
#SBATCH --job-name=puzzle_grid4x_mixed
#SBATCH --partition=a40,rtxpro6k
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-3%4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

CONFIGS=(
  grid4x_sudoku_global_mlp_l1_mixed_rollout_k2
  grid4x_sudoku_global_mlp_l1_mixed_rollout_k4
  grid4x_sudoku_global_mlp_l2_mixed_hierarchy_span4
  grid4x_sudoku_global_mlp_l3_mixed_hierarchy_span4
)

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if (( INDEX < 0 || INDEX >= ${#CONFIGS[@]} )); then
  echo "Unknown SLURM_ARRAY_TASK_ID=${INDEX}; valid range is 0-$(( ${#CONFIGS[@]} - 1 ))" >&2
  exit 2
fi

python -m puzzle_jepa.train.grid0 --config-name "${CONFIGS[$INDEX]}"
