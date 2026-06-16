#!/usr/bin/env bash
#SBATCH --job-name=grid_goal_plan
#SBATCH --partition=a100,a40,rtxpro6k
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-11%12
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

ABLATIONS=(
  M0_full
  R1_no_context_masks
  R2_mean_pooled_distance
  R3_k1_only
  R3_k4
  R3_k8
  R3_k16
  R4_no_goal_nce
  R5_no_progress_rank
  R6_no_action_rank
  R7_no_terminal_corrupt
  R8_no_sigreg
)

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if (( INDEX < 0 || INDEX >= ${#ABLATIONS[@]} )); then
  echo "Unknown SLURM_ARRAY_TASK_ID=${INDEX}; valid range is 0-$(( ${#ABLATIONS[@]} - 1 ))" >&2
  exit 2
fi

ABLATION="${ABLATIONS[$INDEX]}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/grid_goal_sudoku_${ABLATION}"
CHECKPOINT="${RUN_ROOT}/checkpoint.pt"
OUTPUT_DIR="${RUN_ROOT}/planner_eval"

if [[ ! -s "${CHECKPOINT}" ]]; then
  echo "Missing checkpoint ${CHECKPOINT}; cannot run planner eval for ${ABLATION}." >&2
  exit 3
fi

python -m puzzle_jepa.eval.grid_goal_planner_matrix \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --examples "${EXAMPLES:-16}" \
  --max-steps "${MAX_STEPS:-81}" \
  --beam-widths "${BEAM_WIDTHS:-1,4,16,64}" \
  --beam-depths "${BEAM_DEPTHS:-8,16,32,64}" \
  --scores "${SCORES:-oracle_goal_distance,predicted_goal_distance}" \
  --transitions "${TRANSITIONS:-symbolic_reencode,latent_rollout}"
