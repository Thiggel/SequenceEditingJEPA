#!/usr/bin/env bash
#SBATCH --job-name=grid_goal_fu_eval
#SBATCH --partition=rtxpro6k,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-17%12
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

VARIANTS=(
  F0_dense_k16
  F1_dense_k32_detach8
  H0_hierarchy_l4_l16
  H1_hierarchy_dense_l4_l16
  S0_scale_d384_dense
  S1_deeper_d256_dense
)
PLANNER_GROUPS=(beam categorical_cem hierarchical_cem)

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
TOTAL=$(( ${#VARIANTS[@]} * ${#PLANNER_GROUPS[@]} ))
if (( INDEX < 0 || INDEX >= TOTAL )); then
  echo "Unknown SLURM_ARRAY_TASK_ID=${INDEX}; valid range is 0-$(( TOTAL - 1 ))" >&2
  exit 2
fi

planner_index=$(( INDEX % ${#PLANNER_GROUPS[@]} ))
variant_index=$(( INDEX / ${#PLANNER_GROUPS[@]} ))
VARIANT="${VARIANTS[$variant_index]}"
PLANNER_GROUP="${PLANNER_GROUPS[$planner_index]}"

RUN_NAME="grid_goal_followup_${VARIANT}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/grid_goal_followups/${RUN_NAME}"
CHECKPOINT="${RUN_ROOT}/checkpoint.pt"

if [[ ! -s "${CHECKPOINT}" ]]; then
  echo "Missing checkpoint ${CHECKPOINT}; cannot run follow-up eval for ${RUN_NAME}." >&2
  exit 3
fi

case "${PLANNER_GROUP}" in
  beam)
    PLANNERS="mpc_beam"
    EXAMPLES_DEFAULT=10
    OUTPUT_DIR="${RUN_ROOT}/planner_eval_beam"
    ;;
  categorical_cem)
    PLANNERS="categorical_cem"
    EXAMPLES_DEFAULT=10
    OUTPUT_DIR="${RUN_ROOT}/planner_eval_categorical_cem"
    ;;
  hierarchical_cem)
    if [[ "${VARIANT}" != H* ]]; then
      echo "Skipping hierarchical CEM for non-hierarchical variant ${VARIANT}."
      exit 0
    fi
    PLANNERS="hierarchical_cem"
    EXAMPLES_DEFAULT=10
    OUTPUT_DIR="${RUN_ROOT}/planner_eval_hierarchical_cem"
    ;;
  *)
    echo "Unknown planner group ${PLANNER_GROUP}" >&2
    exit 2
    ;;
esac

python -m puzzle_jepa.eval.grid_goal_planner_matrix \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --examples "${EXAMPLES:-$EXAMPLES_DEFAULT}" \
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
  --high-cem-samples "${HIGH_CEM_SAMPLES:-64}" \
  --high-cem-iters "${HIGH_CEM_ITERS:-3}" \
  --high-cem-elites "${HIGH_CEM_ELITES:-8}" \
  --high-cem-momentum "${HIGH_CEM_MOMENTUM:-0.7}" \
  --high-cem-std "${HIGH_CEM_STD:-1.0}" \
  --skip-diagnostics
