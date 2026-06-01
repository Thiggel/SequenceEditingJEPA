#!/usr/bin/env bash
#SBATCH --job-name=puzzle_diag4a_subcem
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
  --output-dir "$RUN_ROOT/diagnostics_subgoal_cem_l${LEVEL}" \
  --rank-examples 0 \
  --drift-examples 0 \
  --trace-examples 0 \
  --planning-examples 0 \
  --cem-examples 0 \
  --subgoal-cem-examples 32 \
  --subgoal-hierarchy-level "$LEVEL" \
  --subgoal-macro-horizon 3 \
  --subgoal-high-population 256 \
  --subgoal-low-population 256 \
  --subgoal-iterations 4 \
  --subgoal-elite-frac 0.2 \
  --subgoal-smoothing 0.7 \
  --subgoal-execute-steps 1 \
  --subgoal-prior-samples 64 \
  --max-unroll-steps 81
