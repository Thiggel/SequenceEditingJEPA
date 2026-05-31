#!/usr/bin/env bash
#SBATCH --job-name=puzzle_diag3d_reset_large
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/sudoku_jepa_5m_local_direct_weighted_rollout_n2"

python -m puzzle_jepa.eval.diagnostics \
  --run-root "$RUN_ROOT" \
  --output-dir "$RUN_ROOT/diagnostics_reset_cadence_large" \
  --rank-examples 0 \
  --drift-examples 0 \
  --trace-examples 0 \
  --planning-examples 128 \
  --max-unroll-steps 81 \
  --horizons 1 2 4 10 20 40 58 \
  --depth-fractions 0.0 0.25 0.5 0.75 \
  --planning-beam-size 4 \
  --planning-branch-size 8 \
  --reset-cadences 4 8
