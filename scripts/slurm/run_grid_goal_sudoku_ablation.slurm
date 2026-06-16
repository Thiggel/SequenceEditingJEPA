#!/usr/bin/env bash
#SBATCH --job-name=grid_goal_train
#SBATCH --partition=a100,a40,rtxpro6k
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-12%13
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
  R9_no_temporal_straightening
)

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if (( INDEX < 0 || INDEX >= ${#ABLATIONS[@]} )); then
  echo "Unknown SLURM_ARRAY_TASK_ID=${INDEX}; valid range is 0-$(( ${#ABLATIONS[@]} - 1 ))" >&2
  exit 2
fi

ABLATION="${ABLATIONS[$INDEX]}"
SEED=$(( 3001 + INDEX ))
RUN_NAME="grid_goal_sudoku_${ABLATION}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/${RUN_NAME}"

python -m puzzle_jepa.train.grid_goal_sudoku \
  --config-name grid_goal_sudoku \
  "seed=${SEED}" \
  "ablation=${ABLATION}" \
  "output_dir=${RUN_ROOT}" \
  "training.learning_rate=1.0e-4"
