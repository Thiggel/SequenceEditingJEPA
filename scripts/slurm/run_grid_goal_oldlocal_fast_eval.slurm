#!/usr/bin/env bash
#SBATCH --job-name=grid_goal_oldlocal_eval
#SBATCH --partition=rtxpro6k,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-17%12
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

VARIANTS=(
  dense_k1
  dense_k4
  dense_k8
  dense_k16
  dense_k32
  hier_l4
  hier_l4_l16
  hier_l4_l16_l32
  hier_l4_l16_shared
  hier_l4_l16_hier_dense
  rank_oracle_progress
  rank_both_progress
  rank_no_progress
  rank_pairwise_oracle_action
  rank_pairwise_both_action
  rank_listwise_pred_action
  rank_listwise_both_action
  rank_no_action
)

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if (( INDEX < 0 || INDEX >= ${#VARIANTS[@]} )); then
  echo "Unknown SLURM_ARRAY_TASK_ID=${INDEX}; valid range is 0-$(( ${#VARIANTS[@]} - 1 ))" >&2
  exit 2
fi

VARIANT="${VARIANTS[$INDEX]}"
RUN_NAME="grid_goal_oldlocal_fast_${VARIANT}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/grid_goal_oldlocal_fast/${RUN_NAME}"
CHECKPOINT="${RUN_ROOT}/checkpoint.pt"
if [[ ! -s "${CHECKPOINT}" ]]; then
  echo "Missing checkpoint ${CHECKPOINT}; cannot evaluate ${RUN_NAME}." >&2
  exit 3
fi

if [[ -z "${PLANNERS:-}" ]]; then
  case "${VARIANT}" in
    hier_*|rank_*)
      PLANNERS="mpc_beam,hierarchical_beam"
      ;;
    *)
      PLANNERS="mpc_beam"
      ;;
  esac
fi

python -m puzzle_jepa.eval.grid_goal_planner_matrix \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${RUN_ROOT}/${EVAL_OUTPUT_DIR_NAME:-planner_eval_oldlocal_fast}" \
  --examples "${EXAMPLES:-10}" \
  --max-steps "${MAX_STEPS:-81}" \
  --beam-widths "${BEAM_WIDTHS:-16}" \
  --beam-depths "${BEAM_DEPTHS:-1,4,16,32}" \
  --scores "${SCORES:-oracle_goal_raw_mse_distance,predicted_goal_raw_mse_distance,oracle_goal_distance,predicted_goal_distance,oracle_goal_raw_euclidean_distance,predicted_goal_raw_euclidean_distance,oracle_goal_changed_cell_raw_euclidean_distance,predicted_goal_changed_cell_raw_euclidean_distance}" \
  --transitions "${TRANSITIONS:-symbolic_reencode,latent_rollout}" \
  --planners "${PLANNERS}" \
  --skip-diagnostics
