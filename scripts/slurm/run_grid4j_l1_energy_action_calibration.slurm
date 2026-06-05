#!/usr/bin/env bash
#SBATCH --job-name=puzzle_grid4j_energy_cal
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/sudoku_jepa_5m_goal_energy_hwm_l1"

python -m puzzle_jepa.eval.energy_action_calibration \
  --run-root "$RUN_ROOT" \
  --output-dir "$RUN_ROOT/energy_action_calibration_l1_original" \
  --examples 16 \
  --max-steps 5 \
  --qualitative-examples 3 \
  --qualitative-actions 6 \
  --seed 1001
