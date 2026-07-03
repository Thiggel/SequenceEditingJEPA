#!/usr/bin/env bash
#SBATCH --job-name=grid_goal_weekend_oversight
#SBATCH --partition=rtxpro6k,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%A.out
#SBATCH --error=logs/%x-%A.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
source scripts/env.sh
mkdir -p logs

python scripts/oversight/grid_goal_weekend.py \
  --manifest scripts/experiments/grid_goal_weekend_manifest.json \
  --work-root "${PUZZLE_JEPA_WORK_ROOT}" \
  --report-root ../sequence-editing-report \
  --repo-root .
