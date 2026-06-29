#!/usr/bin/env bash
#SBATCH --job-name=grid_goal_h1dbg_eval
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-5%6
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

VARIANTS=(
  K14816_LR1e4
  K14816_LR5e5
  K14816_LR5e4
  K16_LR1e4
  K16_LR5e5
  K16_LR5e4
)

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if (( INDEX < 0 || INDEX >= ${#VARIANTS[@]} )); then
  echo "Unknown SLURM_ARRAY_TASK_ID=${INDEX}; valid range is 0-$(( ${#VARIANTS[@]} - 1 ))" >&2
  exit 2
fi

VARIANT="${VARIANTS[$INDEX]}"
RUN_NAME="${RUN_NAME_PREFIX:-grid_goal_h1_debug_seed5204}_${VARIANT}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/grid_goal_h1_debug/${RUN_NAME}"
CHECKPOINT="${RUN_ROOT}/checkpoint.pt"
if [[ ! -s "${CHECKPOINT}" ]]; then
  echo "Missing checkpoint ${CHECKPOINT}; cannot evaluate ${RUN_NAME}." >&2
  exit 3
fi

python -m puzzle_jepa.eval.grid_goal_planner_matrix \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${RUN_ROOT}/planner_eval_h1_debug" \
  --examples "${EXAMPLES:-10}" \
  --max-steps "${MAX_STEPS:-81}" \
  --beam-widths "${BEAM_WIDTHS:-16}" \
  --beam-depths "${BEAM_DEPTHS:-4,16,32,64}" \
  --scores "${SCORES:-oracle_goal_distance,predicted_goal_distance,oracle_goal_raw_euclidean_distance,predicted_goal_raw_euclidean_distance,oracle_goal_changed_cell_raw_euclidean_distance,predicted_goal_changed_cell_raw_euclidean_distance}" \
  --transitions "${TRANSITIONS:-latent_rollout}" \
  --planners "${PLANNERS:-mpc_beam}" \
  --skip-diagnostics
