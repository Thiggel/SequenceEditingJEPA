#!/usr/bin/env bash
#SBATCH --job-name=grid_goal_weekend_eval
#SBATCH --partition=rtxpro6k,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
source scripts/env.sh
mkdir -p logs

VARIANTS=(
  S0_anchor_olddata
  S1_counterfactual_fill
  S2_counterfactual_edit
  S3_counterfactual_edit_adaln
  S4_counterfactual_edit_oldlocal
  E0_base_cf_edit
  E1_hierarchy_l4_l8_l16
  E2_waypoint_h8
  E3_waypoint_h16
  E4_waypoint_h4_h8_h16
  E5_waypoint_h16_hierarchy
  D0_online_h1_grid
  D0_online_h1_single
  D1_online_ordered_h12345_grid
  D1_online_ordered_h12345_single
  D2_online_set_h12345_grid
  D2_online_set_h12345_single
  D3_hybrid_ema_h1_grid
  D3_hybrid_ema_h1_single
  D4_hybrid_ema_set_h12345_grid
  D4_hybrid_ema_set_h12345_single
  D5_ema_vicreg_no_ldad_grid
  D5_ema_vicreg_no_ldad_single
  V1_asym_hindsight
  V2_iql_quasi
  V3_waypoint_asym_hindsight
  I0_integrated_waypoint_asym
  I1_integrated_waypoint_iql
  I2_integrated_best_delta_if_gate_passes_grid
  I2_integrated_best_delta_if_gate_passes_single
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-${VARIANT_INDEX:-0}}"
VARIANT="${VARIANT:-${VARIANTS[$TASK_ID]}}"
RUN_NAME="grid_goal_weekend_${VARIANT}${RUN_SUFFIX:-}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/grid_goal_weekend/${RUN_NAME}"
CHECKPOINT="${RUN_ROOT}/checkpoint.pt"
OUTPUT_DIR="${RUN_ROOT}/planner_eval_weekend${EVAL_OUTPUT_SUFFIX:-}"
BASE_VARIANT="${VARIANT}"
if [[ "${VARIANT}" == *_single ]]; then
  BASE_VARIANT="${VARIANT%_single}"
elif [[ "${VARIANT}" == *_grid ]]; then
  BASE_VARIANT="${VARIANT%_grid}"
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing checkpoint ${CHECKPOINT}" >&2
  exit 2
fi

ALLOW_OVERWRITE="true"
PLANNERS="${PLANNERS:-mpc_beam}"
SCORES="${SCORES:-oracle_goal_raw_euclidean_distance,predicted_goal_raw_euclidean_distance}"
TRANSITIONS="${TRANSITIONS:-latent_rollout,symbolic_reencode}"
WAYPOINT_HORIZON="${WAYPOINT_HORIZON:-8}"

case "${BASE_VARIANT}" in
  S0_anchor_olddata|S1_counterfactual_fill)
    ALLOW_OVERWRITE="false" ;;
  E1_*)
    PLANNERS="mpc_beam,hierarchical_beam" ;;
  E2_waypoint_h8)
    PLANNERS="waypoint_beam"; SCORES="oracle_waypoint_raw_euclidean_distance,predicted_waypoint_raw_euclidean_distance"; WAYPOINT_HORIZON="8" ;;
  E3_waypoint_h16)
    PLANNERS="waypoint_beam"; SCORES="oracle_waypoint_raw_euclidean_distance,predicted_waypoint_raw_euclidean_distance"; WAYPOINT_HORIZON="16" ;;
  E4_waypoint_h4_h8_h16)
    PLANNERS="waypoint_beam"; SCORES="oracle_waypoint_raw_euclidean_distance,predicted_waypoint_raw_euclidean_distance"; WAYPOINT_HORIZON="${WAYPOINT_HORIZON:-16}" ;;
  E5_waypoint_h16_hierarchy|V3_waypoint_asym_hindsight|I0_integrated_waypoint_asym|I1_integrated_waypoint_iql|I2_integrated_best_delta_if_gate_passes)
    PLANNERS="waypoint_beam,waypoint_hierarchical_beam"; SCORES="oracle_waypoint_raw_euclidean_distance,predicted_waypoint_raw_euclidean_distance"; WAYPOINT_HORIZON="16" ;;
  V1_asym_hindsight|V2_iql_quasi)
    SCORES="oracle_goal_projected_euclidean_distance,predicted_goal_projected_euclidean_distance" ;;
  D*)
    SCORES="oracle_goal_raw_euclidean_distance"; PLANNERS="mpc_beam" ;;
esac

ALLOW_ARGS=()
if [[ "${ALLOW_OVERWRITE}" == "true" ]]; then
  ALLOW_ARGS+=(--allow-overwrite)
fi

python -m puzzle_jepa.eval.grid_goal_planner_matrix \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --examples "${EVAL_EXAMPLES:-8}" \
  --max-steps "${MAX_PLAN_STEPS:-96}" \
  --beam-widths "${BEAM_WIDTHS:-16}" \
  --beam-depths "${BEAM_DEPTHS:-4,16}" \
  --scores "${SCORES}" \
  --transitions "${TRANSITIONS}" \
  --planners "${PLANNERS}" \
  --waypoint-horizon "${WAYPOINT_HORIZON}" \
  "${ALLOW_ARGS[@]}"
