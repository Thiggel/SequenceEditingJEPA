#!/usr/bin/env bash
#SBATCH --job-name=puzzle_grid4w_global_horizon
#SBATCH --partition=a40,rtxpro6k
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --array=0-11%6
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

RUN_SPECS=(
  "sudoku_jepa_5m_global_mlp_l1"
  "sudoku_jepa_5m_global_mlp_l2_span4"
  "sudoku_jepa_5m_global_mlp_l3_span4"
)

SCORE_SPECS=(
  "goal_energy"
  "latent_goal"
)

HORIZON_SPECS=(
  "32"
  "64"
)

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
TOTAL=$(( ${#RUN_SPECS[@]} * ${#SCORE_SPECS[@]} * ${#HORIZON_SPECS[@]} ))
if (( INDEX < 0 || INDEX >= TOTAL )); then
  echo "Unknown SLURM_ARRAY_TASK_ID=${INDEX}; valid range is 0-$(( TOTAL - 1 ))" >&2
  exit 2
fi

RUN_INDEX=$(( INDEX / (${#SCORE_SPECS[@]} * ${#HORIZON_SPECS[@]}) ))
REST=$(( INDEX % (${#SCORE_SPECS[@]} * ${#HORIZON_SPECS[@]}) ))
SCORE_INDEX=$(( REST / ${#HORIZON_SPECS[@]} ))
HORIZON_INDEX=$(( REST % ${#HORIZON_SPECS[@]} ))

RUN_NAME="${RUN_SPECS[$RUN_INDEX]}"
SCORE="${SCORE_SPECS[$SCORE_INDEX]}"
HORIZON="${HORIZON_SPECS[$HORIZON_INDEX]}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/${RUN_NAME}"
OUT_DIR="${RUN_ROOT}/diagnostics_mpc_cem_${SCORE}_h${HORIZON}"

python -m puzzle_jepa.eval.diagnostics \
  --run-root "$RUN_ROOT" \
  --output-dir "$OUT_DIR" \
  --rank-examples 0 \
  --drift-examples 0 \
  --trace-examples 0 \
  --planning-examples 0 \
  --cem-examples 0 \
  --mcts-examples 0 \
  --max-unroll-steps 81 \
  --seed $((6200 + INDEX)) \
  --mpc-cem-examples 64 \
  --mpc-cem-score "$SCORE" \
  --mpc-cem-horizon "$HORIZON" \
  --mpc-cem-execute-steps 1 \
  --mpc-cem-population 128 \
  --mpc-cem-iterations 4 \
  --mpc-cem-elite-frac 0.2 \
  --mpc-cem-smoothing 0.7
