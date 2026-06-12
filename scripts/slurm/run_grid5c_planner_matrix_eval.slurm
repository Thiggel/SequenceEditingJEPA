#!/bin/bash
#SBATCH --job-name=puzzle_grid5c_plan
#SBATCH --output=logs/puzzle_grid5c_plan_%A_%a.out
#SBATCH --error=logs/puzzle_grid5c_plan_%A_%a.err
#SBATCH --array=0-11%12
#SBATCH --partition=a40,a100,rtxpro6k
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00

set -euo pipefail

cd /home/hpc/c107fa/c107fa12/sequence-editing
source ./scripts/env.sh
mkdir -p logs

NAMES=(
  "canonical_sigreg_k4"
  "canonical_ema_sigreg_k4"
  "canonical_vicreg_k4"
  "canonical_ema_vicreg_k4"
  "canonical_sigreg_k1"
  "canonical_ema_sigreg_k1"
  "delta_sigreg_k4"
  "mlp_pred_sigreg_k4"
  "mlp_enc_sigreg_k4"
  "oldbest_scaled_sigreg_k4"
  "oldbest_scaled_ema_sigreg_k4"
  "oldbest_scaled_sigreg_k1"
)

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if (( INDEX < 0 || INDEX >= ${#NAMES[@]} )); then
  echo "Unknown SLURM_ARRAY_TASK_ID=${INDEX}; valid range is 0-$(( ${#NAMES[@]} - 1 ))" >&2
  exit 2
fi

RUN_NAME="grid5b_10m_${NAMES[$INDEX]}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/${RUN_NAME}"

python -m puzzle_jepa.eval.grid5_planner_matrix \
  --run-root "${RUN_ROOT}" \
  --output-dir "${RUN_ROOT}/diagnostics_planner_matrix" \
  --seed "$(( 12100 + INDEX ))" \
  --plan-examples 2 \
  --horizons 8 16 32 64 \
  --optimizers beam mcts nn_cem \
  --transition-modes symbolic_reencode latent_rollout \
  --score-modes latent_goal goal_energy \
  --action-mode mutable_overwrite \
  --max-steps 64 \
  --beam-width 4 \
  --branch-size 8 \
  --mcts-simulations 64 \
  --mcts-exploration 1.0 \
  --nn-cem-candidates 64 \
  --nn-cem-elites 8 \
  --nn-cem-iterations 4 \
  --nn-cem-smoothing 0.7
