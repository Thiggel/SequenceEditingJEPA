#!/usr/bin/env bash
#SBATCH --job-name=motion_watch
#SBATCH --partition=a100mig
#SBATCH --gres=gpu:a100small:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source scripts/env.sh

python scripts/oversight/moving_objects.py \
  --manifest "${MOTION_MANIFEST:?MOTION_MANIFEST is required}" \
  --run-root "${PUZZLE_JEPA_WORK_ROOT}/runs/moving_objects"
