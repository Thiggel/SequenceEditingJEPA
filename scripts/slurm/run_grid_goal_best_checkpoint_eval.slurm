#!/usr/bin/env bash
#SBATCH --job-name=grid_goal_ckpt_eval
#SBATCH --partition=rtxpro6k,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-9%10
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

CHECKPOINT_STEPS=(20000 30000 40000 50000 60000)
PLANNER_GROUPS=(beam categorical_cem)

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
TOTAL=$(( ${#CHECKPOINT_STEPS[@]} * ${#PLANNER_GROUPS[@]} ))
if (( INDEX < 0 || INDEX >= TOTAL )); then
  echo "Unknown SLURM_ARRAY_TASK_ID=${INDEX}; valid range is 0-$(( TOTAL - 1 ))" >&2
  exit 2
fi

planner_index=$(( INDEX % ${#PLANNER_GROUPS[@]} ))
step_index=$(( INDEX / ${#PLANNER_GROUPS[@]} ))
STEP="${CHECKPOINT_STEPS[$step_index]}"
PLANNER_GROUP="${PLANNER_GROUPS[$planner_index]}"

RUN_NAME="${RUN_NAME:-grid_goal_action_R4_no_goal_nce_A6_affected_marker_delta_S4_ema_vicreg_D0_uniform}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/grid_goal_action_suite/${RUN_NAME}"
CHECKPOINT="${RUN_ROOT}/checkpoint-${STEP}.pt"
if [[ "${STEP}" == "60000" && ! -s "${CHECKPOINT}" ]]; then
  CHECKPOINT="${RUN_ROOT}/checkpoint.pt"
fi
if [[ ! -s "${CHECKPOINT}" ]]; then
  echo "Missing checkpoint ${CHECKPOINT}; cannot run checkpoint-time eval." >&2
  exit 3
fi

case "${PLANNER_GROUP}" in
  beam)
    PLANNERS="mpc_beam"
    OUTPUT_DIR="${RUN_ROOT}/planner_eval_checkpoint_${STEP}_beam"
    ;;
  categorical_cem)
    PLANNERS="categorical_cem"
    OUTPUT_DIR="${RUN_ROOT}/planner_eval_checkpoint_${STEP}_categorical_cem"
    ;;
  *)
    echo "Unknown planner group ${PLANNER_GROUP}" >&2
    exit 2
    ;;
esac

python -m puzzle_jepa.eval.grid_goal_planner_matrix \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --examples "${EXAMPLES:-10}" \
  --max-steps "${MAX_STEPS:-81}" \
  --beam-widths "${BEAM_WIDTHS:-16}" \
  --beam-depths "${BEAM_DEPTHS:-4,16,32,64}" \
  --scores "${SCORES:-oracle_goal_distance,predicted_goal_distance,oracle_goal_raw_euclidean_distance,predicted_goal_raw_euclidean_distance,oracle_goal_changed_cell_raw_euclidean_distance,predicted_goal_changed_cell_raw_euclidean_distance}" \
  --transitions "${TRANSITIONS:-latent_rollout}" \
  --planners "${PLANNERS}" \
  --cem-samples "${CEM_SAMPLES:-64}" \
  --cem-iters "${CEM_ITERS:-3}" \
  --cem-elites "${CEM_ELITES:-8}" \
  --cem-momentum "${CEM_MOMENTUM:-0.7}" \
  --skip-diagnostics
