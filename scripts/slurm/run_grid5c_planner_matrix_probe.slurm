#!/bin/bash
#SBATCH --job-name=puzzle_grid5c_probe
#SBATCH --output=logs/puzzle_grid5c_probe_%j.out
#SBATCH --error=logs/puzzle_grid5c_probe_%j.err
#SBATCH --partition=a40,a100,rtxpro6k
#SBATCH --exclude=a2143
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

set -euo pipefail

cd /home/hpc/c107fa/c107fa12/sequence-editing
source ./scripts/env.sh
mkdir -p logs

RUN_NAME="grid5b_10m_canonical_ema_vicreg_k4"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/${RUN_NAME}"
OUTPUT_DIR="${RUN_ROOT}/diagnostics_planner_matrix_probe_20260613"

python -m puzzle_jepa.eval.grid5_planner_matrix \
  --run-root "${RUN_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --seed 12613 \
  --plan-examples 1 \
  --horizons 8 \
  --optimizers beam mcts nn_cem \
  --transition-modes symbolic_reencode latent_rollout \
  --score-modes latent_goal goal_energy \
  --action-mode mutable_overwrite \
  --max-steps 32 \
  --beam-width 2 \
  --branch-size 4 \
  --mcts-simulations 16 \
  --mcts-exploration 1.0 \
  --nn-cem-candidates 16 \
  --nn-cem-elites 4 \
  --nn-cem-iterations 2 \
  --nn-cem-smoothing 0.7
