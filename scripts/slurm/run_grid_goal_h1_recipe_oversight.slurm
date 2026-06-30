#!/usr/bin/env bash
#SBATCH --job-name=grid_goal_h1r_oversight
#SBATCH --partition=rtxpro6k,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

ARGS=()
if [[ "${H1_RECIPE_SUBMIT_NEXT:-0}" == "1" ]]; then
  ARGS+=(--submit-next)
fi
if [[ "${H1_RECIPE_CLEANUP:-0}" == "1" ]]; then
  ARGS+=(--cleanup)
fi

python scripts/oversight/grid_goal_h1_recipe.py "${ARGS[@]}"
