#!/usr/bin/env bash
#SBATCH --job-name=grid_goal_h1w2_eval
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-7%8
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

VARIANTS=(
  chosen_anchor
  chosen_affected_context
  chosen_no_temporal
  chosen_no_progress
  chosen_no_action_rank
  chosen_minimal_aux
  chosen_hier_none
  chosen_hier_l4_l16_l32
)

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if (( INDEX < 0 || INDEX >= ${#VARIANTS[@]} )); then
  echo "Unknown SLURM_ARRAY_TASK_ID=${INDEX}; valid range is 0-$(( ${#VARIANTS[@]} - 1 ))" >&2
  exit 2
fi

VARIANT="${VARIANTS[$INDEX]}"
ACTION_CONDITIONING="${H1_WAVE2_ACTION_CONDITIONING:-affected_marker}"
DYNAMICS_WEIGHTING="${H1_WAVE2_DYNAMICS_WEIGHTING:-uniform}"
RUN_NAME="grid_goal_h1_wave2_${ACTION_CONDITIONING}_${DYNAMICS_WEIGHTING}_${VARIANT}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/grid_goal_h1_wave2/${RUN_NAME}"
CHECKPOINT="${RUN_ROOT}/checkpoint.pt"
if [[ ! -s "${CHECKPOINT}" ]]; then
  echo "Missing checkpoint ${CHECKPOINT}; cannot evaluate ${RUN_NAME}." >&2
  exit 3
fi

if [[ "${VARIANT}" == "chosen_hier_none" ]]; then
  DEFAULT_PLANNERS="mpc_beam"
else
  DEFAULT_PLANNERS="mpc_beam,hierarchical_beam"
fi

python -m puzzle_jepa.eval.grid_goal_planner_matrix \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${RUN_ROOT}/${EVAL_OUTPUT_DIR_NAME:-planner_eval_h1_wave2}" \
  --examples "${EXAMPLES:-10}" \
  --max-steps "${MAX_STEPS:-81}" \
  --beam-widths "${BEAM_WIDTHS:-16}" \
  --beam-depths "${BEAM_DEPTHS:-4,16,32,64}" \
  --scores "${SCORES:-oracle_goal_distance,predicted_goal_distance,oracle_goal_raw_euclidean_distance,predicted_goal_raw_euclidean_distance,oracle_goal_raw_mse_distance,predicted_goal_raw_mse_distance,oracle_goal_changed_cell_raw_euclidean_distance,predicted_goal_changed_cell_raw_euclidean_distance,oracle_goal_affected_context_raw_euclidean_distance,predicted_goal_affected_context_raw_euclidean_distance}" \
  --transitions "${TRANSITIONS:-symbolic_reencode,latent_rollout}" \
  --planners "${PLANNERS:-${DEFAULT_PLANNERS}}" \
  --skip-diagnostics
