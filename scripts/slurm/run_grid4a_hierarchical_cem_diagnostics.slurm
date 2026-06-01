#!/usr/bin/env bash
#SBATCH --job-name=puzzle_diag4a_hcem
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-1

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

RUNS=(
  sudoku_jepa_5m_goal_energy_hwm_l2_span9
  sudoku_jepa_5m_goal_energy_hwm_l3_span3
)
LEVELS=(1 2)

RUN_NAME="${RUNS[$SLURM_ARRAY_TASK_ID]}"
LEVEL="${LEVELS[$SLURM_ARRAY_TASK_ID]}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/${RUN_NAME}"

python -m puzzle_jepa.eval.diagnostics \
  --run-root "$RUN_ROOT" \
  --output-dir "$RUN_ROOT/diagnostics_cem_hier_l${LEVEL}" \
  --rank-examples 0 \
  --drift-examples 0 \
  --trace-examples 0 \
  --planning-examples 0 \
  --cem-examples 32 \
  --cem-population 256 \
  --cem-elite-frac 0.2 \
  --cem-iterations 4 \
  --cem-smoothing 0.7 \
  --cem-score hierarchical_latent_goal \
  --cem-hierarchy-level "$LEVEL" \
  --max-unroll-steps 81
