#!/usr/bin/env bash
#SBATCH --job-name=puzzle_grid0
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-1

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

CONFIGS=(
  grid0_sudoku_jepa_5m_oracle_smoke
  grid0_maze_jepa_5m_oracle_smoke
)

CONFIG_NAME="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
python -m puzzle_jepa.train.grid0 --config-name "$CONFIG_NAME"
