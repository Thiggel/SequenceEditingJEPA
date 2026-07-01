#!/usr/bin/env bash
#SBATCH --job-name=grid_goal_minaux_eval
#SBATCH --partition=rtxpro6k,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --array=0-28%29
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

VARIANTS=(
  base
  reg_vicreg
  reg_sigreg
  reg_no_ema
  reg_vicreg_no_ema
  reg_sigreg_no_ema
  reg_vicreg_sigreg
  reg_vicreg_sigreg_no_ema
  rank_pairwise_pred_action
  rank_listwise_pred_action
  rank_pairwise_oracle_action
  rank_listwise_oracle_action
  geom_temporal
  geom_pred_progress
  geom_oracle_progress
  dense_k1
  dense_k2
  dense_k4
  dense_k8
  dense_k16
  hier_none
  hier_l4
  hier_l16
  hier_l4_l16
  hier_l4_l16_l32
  goal_initial_current
  goal_no_stopgrad
  goal_initial_current_no_stopgrad
  goal_distance_field_distill
)

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if (( INDEX < 0 || INDEX >= ${#VARIANTS[@]} )); then
  echo "Unknown SLURM_ARRAY_TASK_ID=${INDEX}; valid range is 0-$(( ${#VARIANTS[@]} - 1 ))" >&2
  exit 2
fi

VARIANT="${VARIANTS[$INDEX]}"
RUN_NAME="grid_goal_minaux_wave_${VARIANT}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/grid_goal_minaux_wave/${RUN_NAME}"
CHECKPOINT="${RUN_ROOT}/checkpoint.pt"
if [[ ! -s "${CHECKPOINT}" ]]; then
  echo "Missing checkpoint ${CHECKPOINT}; cannot evaluate ${RUN_NAME}." >&2
  exit 3
fi

if [[ "${VARIANT}" == "hier_none" ]]; then
  DEFAULT_PLANNERS="mpc_beam"
else
  DEFAULT_PLANNERS="mpc_beam,hierarchical_beam"
fi

python -m puzzle_jepa.eval.grid_goal_planner_matrix \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${RUN_ROOT}/${EVAL_OUTPUT_DIR_NAME:-planner_eval_minaux_wave_fast}" \
  --examples "${EXAMPLES:-8}" \
  --max-steps "${MAX_STEPS:-81}" \
  --beam-widths "${BEAM_WIDTHS:-16}" \
  --beam-depths "${BEAM_DEPTHS:-4,16}" \
  --scores "${SCORES:-oracle_goal_distance,predicted_goal_distance,oracle_goal_raw_euclidean_distance,predicted_goal_raw_euclidean_distance}" \
  --transitions "${TRANSITIONS:-latent_rollout}" \
  --planners "${PLANNERS:-${DEFAULT_PLANNERS}}" \
  --skip-diagnostics
