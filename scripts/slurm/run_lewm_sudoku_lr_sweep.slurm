#!/usr/bin/env bash
#SBATCH --job-name=lewm_sudoku_lr
#SBATCH --partition=a100,a40,rtxpro6k
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-24%12
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

LRS=(
  1e-6 2e-6 3e-6 4e-6 5e-6 6e-6 7e-6 8e-6 9e-6
  1e-5 2e-5 3e-5 4e-5 5e-5 6e-5 7e-5 8e-5 9e-5
  1e-4 2e-4 3e-4 4e-4 5e-4 6e-4 7e-4
)

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if (( INDEX < 0 || INDEX >= ${#LRS[@]} )); then
  echo "Unknown SLURM_ARRAY_TASK_ID=${INDEX}; valid range is 0-$(( ${#LRS[@]} - 1 ))" >&2
  exit 2
fi

LR="${LRS[$INDEX]}"
SEED=$(( 2001 + INDEX ))
RUN_NAME="lewm_sudoku_lr_${LR}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/${RUN_NAME}"

python -m puzzle_jepa.train.lewm_sudoku \
  --config-name lewm_sudoku \
  "seed=${SEED}" \
  "output_dir=${RUN_ROOT}" \
  "training.learning_rate=${LR}"
