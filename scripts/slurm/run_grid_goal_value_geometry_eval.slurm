#!/usr/bin/env bash
#SBATCH --job-name=grid_goal_value_eval
#SBATCH --partition=rtxpro6k,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

VARIANTS=(
  V0_base
  V1_hindsight_metric
  V2_iql_euclidean
  V3_iql_quasimetric
  V4_terminal_value
  V5_success_vector
  V6_success_vector_iql
  V7_success_vector_q
  V8_bad_state_iql_quasi
)

if [[ -n "${VARIANT:-}" ]]; then
  SELECTED_VARIANT="${VARIANT}"
else
  INDEX="${SLURM_ARRAY_TASK_ID:-0}"
  if (( INDEX < 0 || INDEX >= ${#VARIANTS[@]} )); then
    echo "Unknown SLURM_ARRAY_TASK_ID=${INDEX}; valid range is 0-$(( ${#VARIANTS[@]} - 1 ))" >&2
    exit 2
  fi
  SELECTED_VARIANT="${VARIANTS[$INDEX]}"
fi

case "${SELECTED_VARIANT}" in
  V0_base)
    DEFAULT_SCORES="oracle_goal_raw_euclidean_distance,predicted_goal_raw_euclidean_distance"
    ;;
  V1_hindsight_metric|V2_iql_euclidean|V3_iql_quasimetric|V8_bad_state_iql_quasi)
    DEFAULT_SCORES="oracle_goal_projected_euclidean_distance,predicted_goal_projected_euclidean_distance"
    ;;
  V4_terminal_value)
    DEFAULT_SCORES="terminal_value"
    ;;
  V5_success_vector|V6_success_vector_iql|V7_success_vector_q)
    DEFAULT_SCORES="success_metric_distance"
    ;;
  *)
    echo "Unknown variant ${SELECTED_VARIANT}" >&2
    exit 2
    ;;
esac

RUN_SUFFIX="${RUN_SUFFIX:-}"
RUN_NAME="grid_goal_value_geometry_${SELECTED_VARIANT}${RUN_SUFFIX}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/grid_goal_value_geometry/${RUN_NAME}"
CHECKPOINT="${RUN_ROOT}/checkpoint.pt"
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi

EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${RUN_ROOT}/planner_eval_value${EVAL_OUTPUT_SUFFIX:-}}"

python -m puzzle_jepa.eval.grid_goal_planner_matrix \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${EVAL_OUTPUT_DIR}" \
  --examples "${EVAL_EXAMPLES:-8}" \
  --beam-widths "${BEAM_WIDTHS:-16}" \
  --beam-depths "${BEAM_DEPTHS:-1,2,4,16}" \
  --scores "${SCORES:-${DEFAULT_SCORES}}" \
  --transitions "${TRANSITIONS:-latent_rollout,symbolic_reencode}" \
  --planners "${PLANNERS:-mpc_beam}" \
  --seed "${EVAL_SEED:-7401}" \
  --skip-diagnostics
