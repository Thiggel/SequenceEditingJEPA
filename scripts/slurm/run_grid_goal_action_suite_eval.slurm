#!/usr/bin/env bash
#SBATCH --job-name=grid_goal_act_eval
#SBATCH --partition=rtxpro6k,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --array=0-95%32
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

BASE_ABLATIONS=(R4_no_goal_nce R7_no_terminal_corrupt)
ACTION_VARIANTS=(
  A0_action_token
  A1_affected_marker
  A2_local_action_feature
  A3_action_cross_attention
  A4_adaln_action
  A5_action_token_delta
  A6_affected_marker_delta
  A7_local_action_feature_delta
)
STABILITY_VARIANTS=(S0_sigreg S3_ema_sigreg S4_ema_vicreg)
DYNAMICS_VARIANTS=(D0_uniform D1_affected)

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
TOTAL=$(( ${#BASE_ABLATIONS[@]} * ${#ACTION_VARIANTS[@]} * ${#STABILITY_VARIANTS[@]} * ${#DYNAMICS_VARIANTS[@]} ))
if (( INDEX < 0 || INDEX >= TOTAL )); then
  echo "Unknown SLURM_ARRAY_TASK_ID=${INDEX}; valid range is 0-$(( TOTAL - 1 ))" >&2
  exit 2
fi

dyn_index=$(( INDEX % ${#DYNAMICS_VARIANTS[@]} ))
tmp=$(( INDEX / ${#DYNAMICS_VARIANTS[@]} ))
stab_index=$(( tmp % ${#STABILITY_VARIANTS[@]} ))
tmp=$(( tmp / ${#STABILITY_VARIANTS[@]} ))
action_index=$(( tmp % ${#ACTION_VARIANTS[@]} ))
base_index=$(( tmp / ${#ACTION_VARIANTS[@]} ))

ABLATION="${BASE_ABLATIONS[$base_index]}"
ACTION_VARIANT="${ACTION_VARIANTS[$action_index]}"
STABILITY_VARIANT="${STABILITY_VARIANTS[$stab_index]}"
DYNAMICS_VARIANT="${DYNAMICS_VARIANTS[$dyn_index]}"

RUN_NAME="grid_goal_action_${ABLATION}_${ACTION_VARIANT}_${STABILITY_VARIANT}_${DYNAMICS_VARIANT}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/grid_goal_action_suite/${RUN_NAME}"
CHECKPOINT="${RUN_ROOT}/checkpoint.pt"
OUTPUT_DIR="${RUN_ROOT}/planner_eval_latent${OUTPUT_SUFFIX:-}"

if [[ ! -s "${CHECKPOINT}" ]]; then
  echo "Missing checkpoint ${CHECKPOINT}; cannot run planner eval for ${RUN_NAME}." >&2
  exit 3
fi

python -m puzzle_jepa.eval.grid_goal_planner_matrix \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --examples "${EXAMPLES:-10}" \
  --max-steps "${MAX_STEPS:-81}" \
  --beam-widths "${BEAM_WIDTHS:-16}" \
  --beam-depths "${BEAM_DEPTHS:-4,16,32}" \
  --scores "${SCORES:-oracle_goal_distance,predicted_goal_distance,oracle_goal_raw_euclidean_distance,predicted_goal_raw_euclidean_distance,oracle_goal_changed_cell_raw_euclidean_distance,predicted_goal_changed_cell_raw_euclidean_distance}" \
  --transitions "${TRANSITIONS:-latent_rollout}" \
  --skip-diagnostics
