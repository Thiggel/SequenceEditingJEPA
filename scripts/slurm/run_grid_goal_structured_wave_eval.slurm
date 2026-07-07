#!/usr/bin/env bash
#SBATCH --job-name=grid_goal_struct_eval
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

if [[ -z "${VARIANT:-}" ]]; then
  echo "VARIANT must be set." >&2
  exit 2
fi

RUN_SUFFIX="${RUN_SUFFIX:-}"
OUTPUT_ROOT="${PUZZLE_JEPA_WORK_ROOT:-${WORK:?WORK must be set}/sequence-editing-structured-20260707}"
RUN_NAME="grid_goal_structured_wave_${VARIANT}${RUN_SUFFIX}"
RUN_ROOT="${OUTPUT_ROOT}/runs/grid_goal_structured_wave/${RUN_NAME}"
CHECKPOINT="${RUN_ROOT}/checkpoint.pt"
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi

DEFAULT_SCORES="oracle_goal_raw_euclidean_distance"
DEFAULT_PLANNERS="mpc_beam"
DEFAULT_TRANSITIONS="latent_rollout,symbolic_reencode"
DEFAULT_BEAM_DEPTHS="4,16"

case "${VARIANT}" in
  GW0_terminal_goal)
    DEFAULT_SCORES="oracle_goal_raw_euclidean_distance,predicted_goal_raw_euclidean_distance" ;;
  GW1_waypoint_only)
    DEFAULT_SCORES="oracle_waypoint_raw_euclidean_distance,predicted_waypoint_raw_euclidean_distance"
    DEFAULT_PLANNERS="waypoint_beam"
    DEFAULT_BEAM_DEPTHS="4,16" ;;
  GW2_waypoint_goal|GW3_goal_conditioned_waypoint|GW4_multi_waypoint)
    DEFAULT_SCORES="oracle_waypoint_raw_euclidean_distance,predicted_waypoint_raw_euclidean_distance,predicted_waypoint_goal_raw_euclidean_distance"
    DEFAULT_PLANNERS="waypoint_beam"
    DEFAULT_BEAM_DEPTHS="4,16" ;;
  C4*|C5*|C6*)
    DEFAULT_SCORES="oracle_waypoint_raw_euclidean_distance,predicted_waypoint_raw_euclidean_distance,predicted_waypoint_goal_raw_euclidean_distance"
    DEFAULT_PLANNERS="waypoint_beam"
    DEFAULT_BEAM_DEPTHS="4,16" ;;
  PR*|SD*|DJ*|S*|C*)
    DEFAULT_SCORES="oracle_goal_raw_euclidean_distance,oracle_goal_changed_cell_raw_euclidean_distance" ;;
esac

EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${RUN_ROOT}/planner_eval_structured${EVAL_OUTPUT_SUFFIX:-}}"
mkdir -p "${EVAL_OUTPUT_DIR}"

if [[ "${SKIP_PROBES:-0}" != "1" ]]; then
  python -m puzzle_jepa.eval.grid_goal_diagnostics \
    --checkpoint "${CHECKPOINT}" \
    --output-dir "${EVAL_OUTPUT_DIR}/diagnostics" \
    --examples "${DIAGNOSTIC_EXAMPLES:-16}" \
    --seed "${EVAL_SEED:-9707}"
fi

python -m puzzle_jepa.eval.grid_goal_planner_matrix \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${EVAL_OUTPUT_DIR}" \
  --examples "${EVAL_EXAMPLES:-8}" \
  --beam-widths "${BEAM_WIDTHS:-16}" \
  --beam-depths "${BEAM_DEPTHS:-${DEFAULT_BEAM_DEPTHS}}" \
  --scores "${SCORES:-${DEFAULT_SCORES}}" \
  --transitions "${TRANSITIONS:-${DEFAULT_TRANSITIONS}}" \
  --planners "${PLANNERS:-${DEFAULT_PLANNERS}}" \
  --max-steps "${MAX_PLAN_STEPS:-96}" \
  --waypoint-horizon "${WAYPOINT_HORIZON:-8}" \
  --seed "${EVAL_SEED:-9707}" \
  --allow-overwrite \
  --skip-diagnostics
