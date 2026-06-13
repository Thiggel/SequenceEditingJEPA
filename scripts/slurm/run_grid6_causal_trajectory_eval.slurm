#!/bin/bash
#SBATCH --job-name=puzzle_grid6_eval
#SBATCH --output=logs/puzzle_grid6_eval_%A_%a.out
#SBATCH --error=logs/puzzle_grid6_eval_%A_%a.err
#SBATCH --array=0-1%2
#SBATCH --partition=a40,a100,rtxpro6k
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00

set -euo pipefail

cd /home/hpc/c107fa/c107fa12/sequence-editing
source ./scripts/env.sh
mkdir -p logs

NAMES=(
  "k1"
  "mh"
)

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if (( INDEX < 0 || INDEX >= ${#NAMES[@]} )); then
  echo "Unknown SLURM_ARRAY_TASK_ID=${INDEX}; valid range is 0-$(( ${#NAMES[@]} - 1 ))" >&2
  exit 2
fi

RUN_NAME="grid6_causal_traj_${NAMES[$INDEX]}_d320"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/${RUN_NAME}"

python -m puzzle_jepa.eval.grid6_planner_matrix \
  --run-root "${RUN_ROOT}" \
  --output-dir "${RUN_ROOT}/diagnostics_planner_matrix" \
  --seed "$(( 14001 + INDEX ))" \
  --examples 2 \
  --planners beam cem mcts \
  --transition-modes symbolic_reencode latent_rollout \
  --score-modes latent_goal goal_energy \
  --single-horizons 4 8 16 \
  --mean-horizons 1 2 4 8 16 \
  --action-mode mutable_overwrite \
  --max-steps 32 \
  --beam-width 3 \
  --branch-size 6 \
  --cem-candidates 48 \
  --cem-elites 8 \
  --cem-iterations 3 \
  --cem-smoothing 0.7 \
  --mcts-simulations 32
