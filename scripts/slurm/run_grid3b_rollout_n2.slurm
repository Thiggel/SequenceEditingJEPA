#!/usr/bin/env bash
#SBATCH --job-name=puzzle_grid3b
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

python -m puzzle_jepa.train.grid0 --config-name grid3b_sudoku_jepa_5m_local_direct_weighted_rollout_n2
