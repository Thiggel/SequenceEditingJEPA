#!/usr/bin/env bash
#SBATCH --job-name=grid_goal_weekend_eval
#SBATCH --partition=rtxpro6k,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
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
EVAL_MODE="${EVAL_MODE:-default}"
OUTPUT_SUFFIX="${EVAL_OUTPUT_SUFFIX:-}"
HIGH_CEM_OPTIMIZER="${HIGH_CEM_OPTIMIZER:-cem}"
HIGH_CEM_CODEBOOK="${HIGH_CEM_CODEBOOK:-none}"
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
    PLANNERS="waypoint_beam"; SCORES="oracle_waypoint_raw_euclidean_distance,predicted_waypoint_raw_euclidean_distance"; WAYPOINT_HORIZON="4" ;;
  E5_waypoint_h16_hierarchy|V3_waypoint_asym_hindsight|I0_integrated_waypoint_asym|I1_integrated_waypoint_iql|I2_integrated_best_delta_if_gate_passes)
    PLANNERS="waypoint_beam,waypoint_hierarchical_cem"; SCORES="oracle_waypoint_raw_euclidean_distance,predicted_waypoint_raw_euclidean_distance"; WAYPOINT_HORIZON="16" ;;
  V1_asym_hindsight|V2_iql_quasi)
    SCORES="oracle_goal_projected_euclidean_distance,predicted_goal_projected_euclidean_distance" ;;
  D*)
    SCORES="oracle_goal_raw_euclidean_distance"; PLANNERS="mpc_beam" ;;
esac

case "${EVAL_MODE}" in
  default)
    ;;
  waypoint_cem)
    PLANNERS="waypoint_hierarchical_cem"
    OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-_waypoint_cem}"
    HIGH_CEM_OPTIMIZER="cem"
    HIGH_CEM_CODEBOOK="${HIGH_CEM_CODEBOOK:-none}"
    ;;
  waypoint_mppi)
    PLANNERS="waypoint_hierarchical_cem"
    OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-_waypoint_mppi}"
    HIGH_CEM_OPTIMIZER="mppi"
    HIGH_CEM_CODEBOOK="${HIGH_CEM_CODEBOOK:-none}"
    ;;
  waypoint_cem_codebook)
    PLANNERS="waypoint_hierarchical_cem"
    OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-_waypoint_cem_codebook}"
    HIGH_CEM_OPTIMIZER="cem"
    HIGH_CEM_CODEBOOK="init"
    ;;
  waypoint_mppi_codebook)
    PLANNERS="waypoint_hierarchical_cem"
    OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-_waypoint_mppi_codebook}"
    HIGH_CEM_OPTIMIZER="mppi"
    HIGH_CEM_CODEBOOK="init"
    ;;
  *)
    echo "Unknown EVAL_MODE=${EVAL_MODE}" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="${RUN_ROOT}/planner_eval_weekend${OUTPUT_SUFFIX}"

ALLOW_ARGS=()
if [[ "${ALLOW_OVERWRITE}" == "true" ]]; then
  ALLOW_ARGS+=(--allow-overwrite)
fi
DIAGNOSTIC_ARGS=()
if [[ "${SKIP_DIAGNOSTICS:-0}" == "1" ]]; then
  DIAGNOSTIC_ARGS+=(--skip-diagnostics)
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
  --cem-samples "${CEM_SAMPLES:-32}" \
  --cem-iters "${CEM_ITERS:-2}" \
  --cem-elites "${CEM_ELITES:-8}" \
  --cem-momentum "${CEM_MOMENTUM:-0.7}" \
  --high-cem-samples "${HIGH_CEM_SAMPLES:-32}" \
  --high-cem-iters "${HIGH_CEM_ITERS:-2}" \
  --high-cem-elites "${HIGH_CEM_ELITES:-8}" \
  --high-cem-momentum "${HIGH_CEM_MOMENTUM:-0.7}" \
  --high-cem-std "${HIGH_CEM_STD:-1.0}" \
  --high-cem-optimizer "${HIGH_CEM_OPTIMIZER}" \
  --high-cem-temperature "${HIGH_CEM_TEMPERATURE:-1.0}" \
  --high-cem-codebook "${HIGH_CEM_CODEBOOK}" \
  --high-cem-codebook-size "${HIGH_CEM_CODEBOOK_SIZE:-64}" \
  "${DIAGNOSTIC_ARGS[@]}" \
  "${ALLOW_ARGS[@]}"
