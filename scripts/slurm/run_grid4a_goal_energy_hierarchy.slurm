#!/usr/bin/env bash
#SBATCH --job-name=puzzle_grid4a
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-2

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

CONFIGS=(
  grid4a_sudoku_jepa_5m_goal_energy_cem_l1
  grid4a_sudoku_jepa_5m_goal_energy_cem_l2
  grid4a_sudoku_jepa_5m_goal_energy_cem_l3
)

CONFIG_NAME="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
python -m puzzle_jepa.train.grid0 --config-name "$CONFIG_NAME"
