#!/usr/bin/env bash
#SBATCH --job-name=grid_goal_clean17_eval
#SBATCH --partition=rtxpro6k,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

VARIANTS=(
  W_uniform_H0_G_none
  W_uniform_H4_G_none
  W_uniform_H4_16_G_none
  W_uniform_H4_16_32_G_none
  W_inv_sqrt_H0_G_none
  W_inv_sqrt_H4_G_none
  W_inv_sqrt_H4_16_G_none
  W_inv_sqrt_H4_16_32_G_none
  W_gamma_H0_G_none
  W_gamma_H4_G_none
  W_gamma_H4_16_G_none
  W_gamma_H4_16_32_G_none
  G_ic_detached
  G_ic_non_detached
  G_ic_online_no_stopgrad
  G_ic_field_plus_mse
  G_ic_field_only
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

RUN_NAME="grid_goal_clean17_${SELECTED_VARIANT}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/grid_goal_clean17/${RUN_NAME}"
CHECKPOINT="${RUN_ROOT}/checkpoint.pt"
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi

PLANNER_MODES="mpc_beam"
case "${SELECTED_VARIANT}" in
  *_H0_*)
    PLANNER_MODES="mpc_beam"
    ;;
  *)
    PLANNER_MODES="mpc_beam,hierarchical_beam"
    ;;
esac

SCORE_MODES="oracle_goal_raw_euclidean_distance"
case "${SELECTED_VARIANT}" in
  G_*)
    SCORE_MODES="oracle_goal_raw_euclidean_distance,predicted_goal_raw_euclidean_distance"
    ;;
esac

python -m puzzle_jepa.eval.grid_goal_planner_matrix \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${RUN_ROOT}/planner_eval_clean17_fast" \
  --examples "${EVAL_EXAMPLES:-8}" \
  --beam-widths "${BEAM_WIDTHS:-16}" \
  --beam-depths "${BEAM_DEPTHS:-4,16}" \
  --scores "${SCORES:-${SCORE_MODES}}" \
  --transitions "${TRANSITIONS:-latent_rollout}" \
  --planners "${PLANNERS:-${PLANNER_MODES}}" \
  --seed "${EVAL_SEED:-5204}"
