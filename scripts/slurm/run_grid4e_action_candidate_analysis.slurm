#!/usr/bin/env bash
#SBATCH --job-name=puzzle_grid4e_action_rank
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-6

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

RUNS=(
  sudoku_jepa_5m_goal_energy_hwm_l1
  sudoku_jepa_5m_goal_energy_contrast_l1_nce
  sudoku_jepa_5m_goal_energy_contrast_l1_infonce
  sudoku_jepa_5m_goal_energy_contrast_l1_margin
  sudoku_jepa_5m_goal_energy_contrast_l1_nce_mono
  sudoku_jepa_5m_goal_energy_contrast_l1_infonce_mono
  sudoku_jepa_5m_goal_energy_contrast_l1_margin_mono
)

RUN_NAME="${RUNS[$SLURM_ARRAY_TASK_ID]}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/${RUN_NAME}"

python -m puzzle_jepa.eval.action_candidate_analysis \
  --run-root "$RUN_ROOT" \
  --output-dir "$RUN_ROOT/action_candidate_analysis_goal_energy" \
  --examples 128 \
  --max-steps 81 \
  --score-mode goal_energy \
  --seed 901
