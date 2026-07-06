#!/usr/bin/env bash
#SBATCH --job-name=grid_goal_verifier_repair_eval
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

RUN_SUFFIX="${RUN_SUFFIX:-_repair}"
OUTPUT_ROOT="${PUZZLE_JEPA_WORK_ROOT:-${WORK:?WORK must be set}/sequence-editing-repair-20260706}"
RUN_NAME="grid_goal_verifier_repair_${VARIANT}${RUN_SUFFIX}"
RUN_ROOT="${OUTPUT_ROOT}/runs/grid_goal_verifier_repair/${RUN_NAME}"
CHECKPOINT="${RUN_ROOT}/checkpoint.pt"
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi

case "${VARIANT}" in
  B0_current_E4|P1_pred_h1|P2_pred_h4|P3_pred_h8|P4_score_consistency)
    DEFAULT_SCORES="verifier_energy,remaining_edit_count,compatibility_energy" ;;
  *)
    DEFAULT_SCORES="verifier_energy,remaining_edit_count,compatibility_energy" ;;
esac

EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${RUN_ROOT}/planner_eval_repair${EVAL_OUTPUT_SUFFIX:-}}"
mkdir -p "${EVAL_OUTPUT_DIR}"

python -m puzzle_jepa.eval.grid_goal_verifier_diagnostics \
  --checkpoint "${CHECKPOINT}" \
  --output "${EVAL_OUTPUT_DIR}/verifier_diagnostics.json" \
  --examples "${DIAGNOSTIC_EXAMPLES:-16}" \
  --max-actions "${DIAGNOSTIC_MAX_ACTIONS:-256}" \
  --seed "${EVAL_SEED:-8801}"

python -m puzzle_jepa.eval.grid_goal_planner_matrix \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${EVAL_OUTPUT_DIR}" \
  --examples "${EVAL_EXAMPLES:-8}" \
  --beam-widths "${BEAM_WIDTHS:-16}" \
  --beam-depths "${BEAM_DEPTHS:-1,4,8}" \
  --scores "${SCORES:-${DEFAULT_SCORES}}" \
  --transitions "${TRANSITIONS:-latent_rollout,symbolic_reencode}" \
  --planners "${PLANNERS:-mpc_beam}" \
  --max-steps "${MAX_PLAN_STEPS:-96}" \
  --seed "${EVAL_SEED:-8801}" \
  --allow-overwrite \
  --skip-diagnostics
