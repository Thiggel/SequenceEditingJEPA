#!/usr/bin/env bash
#SBATCH --job-name=grid_goal_minaux_factor_eval
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
  A_anchor_repro
  A_refactor_equiv_14816
  A_refactor_equiv_14816_dropout_off
  A_smooth_14816_like
  A_uniform_k16
  A_inv_sqrt_k16
  A_gamma_k16
  A_inv_sqrt_k8
  A_old_path_h16_only
  A_old_path_h8_only
  A_no_goal_mse
  A_initial_current_goal
  A_no_hierarchy
  A_no_predict_delta
  A_anchor_dropout_off_fp32
  A_refactor_equiv_14816_dropout_off_fp32
  A_anchor_dropout_off_lr5e5
  A_refactor_equiv_14816_dropout_off_lr5e5
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

RUN_NAME="grid_goal_minaux_factor_${SELECTED_VARIANT}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/grid_goal_minaux_factor/${RUN_NAME}"
CHECKPOINT="${RUN_ROOT}/checkpoint.pt"
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi

PLANNER_MODES="mpc_beam,hierarchical_beam"
case "${SELECTED_VARIANT}" in
  A_no_hierarchy)
    PLANNER_MODES="mpc_beam"
    ;;
esac

python -m puzzle_jepa.eval.grid_goal_planner_matrix \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${RUN_ROOT}/planner_eval_minaux_factor_fast" \
  --examples "${EVAL_EXAMPLES:-8}" \
  --beam-widths "${BEAM_WIDTHS:-16}" \
  --beam-depths "${BEAM_DEPTHS:-4,16}" \
  --scores "${SCORES:-oracle_goal_raw_euclidean_distance,predicted_goal_raw_euclidean_distance}" \
  --transitions "${TRANSITIONS:-latent_rollout}" \
  --planners "${PLANNERS:-${PLANNER_MODES}}" \
  --seed "${EVAL_SEED:-5204}" \
  --skip-diagnostics
