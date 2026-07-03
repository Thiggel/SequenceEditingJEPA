#!/usr/bin/env bash
#SBATCH --job-name=grid_goal_delta_jepa_eval
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
  FB_online_noema_nogoal
  FB_online_noema_goal
  FB_stopgrad_noema_nogoal
  FB_stopgrad_noema_goal
  FB_stopgrad_ema_nogoal
  FB_stopgrad_ema_goal
  SV_online_nogoal
  SV_online_goal
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

RUN_SUFFIX="${RUN_SUFFIX:-}"
RUN_NAME="grid_goal_delta_jepa_${SELECTED_VARIANT}${RUN_SUFFIX}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/grid_goal_delta_jepa/${RUN_NAME}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${RUN_ROOT}/planner_eval_delta_jepa_fast${EVAL_OUTPUT_SUFFIX:-}}"
CHECKPOINT="${RUN_ROOT}/checkpoint.pt"
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi

python -m puzzle_jepa.eval.grid_goal_planner_matrix \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${EVAL_OUTPUT_DIR}" \
  --examples "${EVAL_EXAMPLES:-8}" \
  --beam-widths "${BEAM_WIDTHS:-16}" \
  --beam-depths "${BEAM_DEPTHS:-4,16}" \
  --scores "${SCORES:-oracle_goal_raw_euclidean_distance,predicted_goal_raw_euclidean_distance}" \
  --transitions "${TRANSITIONS:-latent_rollout,symbolic_reencode}" \
  --planners "${PLANNERS:-mpc_beam}" \
  --seed "${EVAL_SEED:-6206}" \
  --skip-diagnostics
