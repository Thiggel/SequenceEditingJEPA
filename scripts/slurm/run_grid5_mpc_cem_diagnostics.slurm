#!/bin/bash
#SBATCH --job-name=puzzle_grid5_mpc_cem
#SBATCH --output=logs/puzzle_grid5_mpc_cem_%A_%a.out
#SBATCH --error=logs/puzzle_grid5_mpc_cem_%A_%a.err
#SBATCH --array=0-23%24
#SBATCH --partition=a40,a100,rtxpro6k
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00

set -euo pipefail

cd /home/hpc/c107fa/c107fa12/sequence-editing
source ./scripts/env.sh
mkdir -p logs

SPECS=(
  "mlp mlp state 32"
  "mlp mlp state 64"
  "mlp mlp state 128"
  "mlp mlp delta 32"
  "mlp mlp delta 64"
  "mlp mlp delta 128"
  "mlp ar_transformer state 32"
  "mlp ar_transformer state 64"
  "mlp ar_transformer state 128"
  "mlp ar_transformer delta 32"
  "mlp ar_transformer delta 64"
  "mlp ar_transformer delta 128"
  "cls_transformer mlp state 32"
  "cls_transformer mlp state 64"
  "cls_transformer mlp state 128"
  "cls_transformer mlp delta 32"
  "cls_transformer mlp delta 64"
  "cls_transformer mlp delta 128"
  "cls_transformer ar_transformer state 32"
  "cls_transformer ar_transformer state 64"
  "cls_transformer ar_transformer state 128"
  "cls_transformer ar_transformer delta 32"
  "cls_transformer ar_transformer delta 64"
  "cls_transformer ar_transformer delta 128"
)

read -r ENCODER PREDICTOR TARGET LATENT <<<"${SPECS[$SLURM_ARRAY_TASK_ID]}"
RUN_NAME="grid5_sigreg_${ENCODER}_${PREDICTOR}_${TARGET}_z${LATENT}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/${RUN_NAME}"
OUT_DIR="${RUN_ROOT}/diagnostics_mpc_cem"

python -m puzzle_jepa.eval.grid5_mpc_cem_diagnostics \
  --run-root "${RUN_ROOT}" \
  --output-dir "${OUT_DIR}" \
  --seed "$((5000 + SLURM_ARRAY_TASK_ID))" \
  --horizons 4 8 16 32 64 \
  --plan-examples 4 \
  --max-mpc-steps 64 \
  --candidates 300 \
  --elites 30 \
  --iterations 10 \
  --qualitative-examples 3 \
  --qualitative-candidates 6 \
  --qualitative-cem-candidates 128 \
  --qualitative-cem-iterations 5
