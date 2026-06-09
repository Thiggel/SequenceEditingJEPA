#!/usr/bin/env bash
#SBATCH --job-name=puzzle_grid4q_recur
#SBATCH --partition=a40,rtxpro6k
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

METHODS=(
  terminal_energy
  terminal_energy
  terminal_energy
  action_advantage
  action_advantage
  action_advantage
  state_value
  state_value
  state_value
  contrastive_margin
  contrastive_margin
  contrastive_margin
)
OPTIMIZERS=(
  cem
  gd
  gd_reachability
  cem
  gd
  gd_reachability
  cem
  gd
  gd_reachability
  cem
  gd
  gd_reachability
)
HIGH_SCORES=(
  goal_energy
  goal_energy
  goal_energy
  latent_goal
  latent_goal
  latent_goal
  goal_value
  goal_value
  goal_value
  goal_energy
  goal_energy
  goal_energy
)

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
METHOD="${METHODS[$INDEX]}"
OPTIMIZER="${OPTIMIZERS[$INDEX]}"
HIGH_SCORE="${HIGH_SCORES[$INDEX]}"
RUN_NAME="sudoku_jepa_5m_hier_value_l3_span4_${METHOD}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/${RUN_NAME}"
OUT_DIR="${RUN_ROOT}/diagnostics_recursive_l2_${HIGH_SCORE}_${OPTIMIZER}"

python -m puzzle_jepa.eval.diagnostics \
  --run-root "$RUN_ROOT" \
  --output-dir "$OUT_DIR" \
  --rank-examples 0 \
  --drift-examples 0 \
  --trace-examples 0 \
  --planning-examples 0 \
  --cem-examples 0 \
  --recursive-subgoal-examples 16 \
  --recursive-hierarchy-level 2 \
  --recursive-macro-horizon 5 \
  --recursive-high-population 96 \
  --recursive-low-population 96 \
  --recursive-iterations 3 \
  --recursive-elite-frac 0.2 \
  --recursive-smoothing 0.7 \
  --recursive-execute-steps 1 \
  --recursive-prior-samples 32 \
  --recursive-high-score "$HIGH_SCORE" \
  --recursive-optimizer "$OPTIMIZER" \
  --recursive-gd-steps 24 \
  --recursive-gd-lr 0.03 \
  --recursive-reachability-weight 0.01 \
  --max-unroll-steps 81 \
  --seed $((4300 + SLURM_ARRAY_TASK_ID))
