#!/usr/bin/env bash
#SBATCH --job-name=grid_goal_oversight
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

ARGS=(--stage "${GRID_GOAL_STAGE:-goal_conditioning}")
if [[ "${OVERSIGHT_SUBMIT_NEXT:-0}" == "1" ]]; then
  ARGS+=(--submit-next)
fi
if [[ "${OVERSIGHT_CLEANUP:-0}" == "1" ]]; then
  ARGS+=(--cleanup)
fi
if [[ "${OVERSIGHT_DELETE_CHECKPOINTS:-0}" == "1" ]]; then
  ARGS+=(--delete-checkpoints)
fi

python scripts/oversight/grid_goal_next_wave.py "${ARGS[@]}"
