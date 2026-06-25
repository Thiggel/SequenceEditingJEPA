#!/usr/bin/env bash
#SBATCH --job-name=grid_goal_next_eval
#SBATCH --partition=rtxpro6k,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-6%4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

STAGE="${GRID_GOAL_STAGE:-goal_conditioning}"
INDEX="${SLURM_ARRAY_TASK_ID:-0}"

case "${STAGE}" in
  goal_conditioning)
    VARIANTS=(G0_context G1_initial_current G2_initial_current_oracle_progress)
    ;;
  dense_horizon)
    VARIANTS=(DK2 DK4 DK8 DK16 DK32)
    ;;
  hierarchy_levels)
    VARIANTS=(H_empty H2 H2_4 H2_4_8 H2_4_8_16 H2_4_8_16_32)
    ;;
  predictor_delta_topk)
    VARIANTS=(P0_separate_delta P1_shared_delta P2_shared_no_delta P3_shared_delta_topk_train)
    ;;
  ranking_losses)
    VARIANTS=(R_pred_progress R_oracle_progress R_both_progress R_no_progress R_pairwise R_listwise R_no_action_rank)
    ;;
  hierarchical_planning)
    VARIANTS=(BASE)
    ;;
  policy_prior)
    VARIANTS=(PP0_no_prior PP1_pairwise_prior PP2_listwise_prior PP3_listwise_prior_strong)
    ;;
  *)
    echo "Unknown GRID_GOAL_STAGE=${STAGE}" >&2
    exit 2
    ;;
esac

if (( INDEX < 0 || INDEX >= ${#VARIANTS[@]} )); then
  echo "Skipping array index ${INDEX} for stage ${STAGE}; variant count is ${#VARIANTS[@]}."
  exit 0
fi

VARIANT="${VARIANTS[$INDEX]}"
RUN_NAME="grid_goal_next_${STAGE}_${VARIANT}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/grid_goal_next_wave/${RUN_NAME}"
CHECKPOINT="${RUN_ROOT}/checkpoint.pt"
if [[ ! -s "${CHECKPOINT}" ]]; then
  echo "Missing checkpoint ${CHECKPOINT}; cannot evaluate ${RUN_NAME}." >&2
  exit 3
fi

BASE_SCORES="oracle_goal_distance,predicted_goal_distance,oracle_goal_changed_cell_raw_euclidean_distance,predicted_goal_changed_cell_raw_euclidean_distance"
DELTA_SCORES="oracle_goal_delta_top1_raw_euclidean_distance,predicted_goal_delta_top1_raw_euclidean_distance,oracle_goal_delta_top3_raw_euclidean_distance,predicted_goal_delta_top3_raw_euclidean_distance,oracle_goal_delta_top5_raw_euclidean_distance,predicted_goal_delta_top5_raw_euclidean_distance"
SCORES="${SCORES:-${BASE_SCORES},${DELTA_SCORES}}"

if [[ -z "${PLANNERS:-}" ]]; then
  PLANNERS="mpc_beam"
  if [[ "${STAGE}" == "dense_horizon" || "${STAGE}" == "predictor_delta_topk" || "${STAGE}" == "hierarchical_planning" || "${STAGE}" == "policy_prior" ]]; then
    PLANNERS="mpc_beam,hierarchical_beam"
  elif [[ "${STAGE}" == "hierarchy_levels" && "${VARIANT}" != "H_empty" ]]; then
    PLANNERS="mpc_beam,hierarchical_beam"
  fi
fi

python -m puzzle_jepa.eval.grid_goal_planner_matrix \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${RUN_ROOT}/planner_eval_next" \
  --examples "${EXAMPLES:-10}" \
  --max-steps "${MAX_STEPS:-81}" \
  --beam-widths "${BEAM_WIDTHS:-16}" \
  --beam-depths "${BEAM_DEPTHS:-4,16,32,64}" \
  --scores "${SCORES}" \
  --transitions "${TRANSITIONS:-latent_rollout}" \
  --planners "${PLANNERS}" \
  --skip-diagnostics
