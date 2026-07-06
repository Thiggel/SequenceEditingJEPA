#!/usr/bin/env bash
#SBATCH --job-name=grid_goal_single_wide_eval
#SBATCH --partition=rtxpro6k,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
source scripts/env.sh
mkdir -p logs

VARIANTS=(
  W0_ema_vicreg_d1024
  W1_ema_ldad_set_d1024
  W2_ldad_vicreg_set_d1024
  W3_ldad_only_set_d1024
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

RUN_NAME="grid_goal_single_wide_${SELECTED_VARIANT}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/grid_goal_single_wide/${RUN_NAME}"
CHECKPOINT="${RUN_ROOT}/checkpoint.pt"
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing checkpoint ${CHECKPOINT}" >&2
  exit 2
fi

DIAGNOSTIC_ARGS=()
if [[ "${SKIP_DIAGNOSTICS:-1}" == "1" ]]; then
  DIAGNOSTIC_ARGS+=(--skip-diagnostics)
fi

python -m puzzle_jepa.eval.grid_goal_planner_matrix \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${RUN_ROOT}/planner_eval_oracle" \
  --examples "${EVAL_EXAMPLES:-8}" \
  --max-steps "${MAX_PLAN_STEPS:-96}" \
  --beam-widths "${BEAM_WIDTHS:-16}" \
  --beam-depths "${BEAM_DEPTHS:-4,16}" \
  --scores "oracle_goal_raw_euclidean_distance" \
  --transitions "latent_rollout,symbolic_reencode" \
  --planners "mpc_beam" \
  --allow-overwrite \
  "${DIAGNOSTIC_ARGS[@]}"
