#!/usr/bin/env bash
#SBATCH --job-name=grid_goal_verifier_eval
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
  E0_base_oracle_sanity
  E1_compat_state
  E2_remaining_state
  E3_wr_state
  E4_wr_predicted
  E5_wr_pairwise_rank
  E6_wr_listwise_rank
  E7_wr_listwise_policy
  E8_wr_no_counterfactual
  E9_wr_no_corruption
  F0_full_score
  F1_full_policy
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
  E0_base_oracle_sanity)
    DEFAULT_SCORES="oracle_goal_raw_euclidean_distance" ;;
  E1_compat_state)
    DEFAULT_SCORES="compatibility_energy" ;;
  E2_remaining_state)
    DEFAULT_SCORES="remaining_edit_count" ;;
  *)
    DEFAULT_SCORES="verifier_energy,compatibility_energy,remaining_edit_count" ;;
esac

RUN_SUFFIX="${RUN_SUFFIX:-}"
RUN_NAME="grid_goal_verifier_energy_${SELECTED_VARIANT}${RUN_SUFFIX}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/grid_goal_verifier_energy/${RUN_NAME}"
CHECKPOINT="${RUN_ROOT}/checkpoint.pt"
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi

EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${RUN_ROOT}/planner_eval_verifier${EVAL_OUTPUT_SUFFIX:-}}"
mkdir -p "${EVAL_OUTPUT_DIR}"

python -m puzzle_jepa.eval.grid_goal_verifier_diagnostics \
  --checkpoint "${CHECKPOINT}" \
  --output "${EVAL_OUTPUT_DIR}/verifier_diagnostics.json" \
  --examples "${DIAGNOSTIC_EXAMPLES:-16}" \
  --max-actions "${DIAGNOSTIC_MAX_ACTIONS:-128}" \
  --seed "${EVAL_SEED:-7801}"

python -m puzzle_jepa.eval.grid_goal_planner_matrix \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${EVAL_OUTPUT_DIR}" \
  --examples "${EVAL_EXAMPLES:-8}" \
  --beam-widths "${BEAM_WIDTHS:-16}" \
  --beam-depths "${BEAM_DEPTHS:-1,4,8,16}" \
  --scores "${SCORES:-${DEFAULT_SCORES}}" \
  --transitions "${TRANSITIONS:-latent_rollout,symbolic_reencode}" \
  --planners "${PLANNERS:-mpc_beam}" \
  --max-steps "${MAX_PLAN_STEPS:-96}" \
  --seed "${EVAL_SEED:-7801}" \
  --allow-overwrite \
  --skip-diagnostics
