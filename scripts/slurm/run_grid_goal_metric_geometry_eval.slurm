#!/usr/bin/env bash
#SBATCH --job-name=grid_goal_metric_eval
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
  FB_M0_goalpred_mse
  FB_M1_terminal_progress_bad
  FB_M2_hindsight_bad
  FB_M3_contrastive_bad
  FB_M4_terminal_progress_asym
  FB_M5_hindsight_asym
  SV_M0_goalpred_mse
  SV_M1_terminal_progress_bad
  SV_M2_hindsight_bad
  SV_M3_contrastive_bad
  SV_M4_terminal_progress_asym
  SV_M5_hindsight_asym
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

SCORE_KIND="${SCORE_KIND:-oracle}"
case "${SCORE_KIND}" in
  oracle) SCORE_MODE="oracle_goal_projected_euclidean_distance" ;;
  predicted) SCORE_MODE="predicted_goal_projected_euclidean_distance" ;;
  both) SCORE_MODE="oracle_goal_projected_euclidean_distance,predicted_goal_projected_euclidean_distance" ;;
  *)
    echo "Unknown SCORE_KIND=${SCORE_KIND}; expected oracle, predicted, or both." >&2
    exit 2
    ;;
esac

RUN_SUFFIX="${RUN_SUFFIX:-}"
RUN_NAME="grid_goal_metric_geometry_${SELECTED_VARIANT}${RUN_SUFFIX}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/grid_goal_metric_geometry/${RUN_NAME}"
CHECKPOINT="${RUN_ROOT}/checkpoint.pt"
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi

EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${RUN_ROOT}/planner_eval_metric_${SCORE_KIND}${EVAL_OUTPUT_SUFFIX:-}}"

python -m puzzle_jepa.eval.grid_goal_planner_matrix \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${EVAL_OUTPUT_DIR}" \
  --examples "${EVAL_EXAMPLES:-8}" \
  --beam-widths "${BEAM_WIDTHS:-16}" \
  --beam-depths "${BEAM_DEPTHS:-4,16}" \
  --scores "${SCORES:-${SCORE_MODE}}" \
  --transitions "${TRANSITIONS:-latent_rollout}" \
  --planners "${PLANNERS:-mpc_beam}" \
  --seed "${EVAL_SEED:-7301}" \
  --skip-diagnostics
