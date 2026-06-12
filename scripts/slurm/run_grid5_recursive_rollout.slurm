#!/bin/bash
#SBATCH --job-name=puzzle_grid5_recursive
#SBATCH --output=logs/puzzle_grid5_recursive_%A_%a.out
#SBATCH --error=logs/puzzle_grid5_recursive_%A_%a.err
#SBATCH --array=0-5%6
#SBATCH --partition=a40,a100,rtxpro6k
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00

set -euo pipefail

cd /home/hpc/c107fa/c107fa12/sequence-editing
source ./scripts/env.sh
mkdir -p logs

SPECS=(
  "mlp 2"
  "mlp 4"
  "mlp 8"
  "ar_transformer 2"
  "ar_transformer 4"
  "ar_transformer 8"
)

read -r PREDICTOR RECURSIVE_K <<<"${SPECS[$SLURM_ARRAY_TASK_ID]}"
ROLLOUT_STEPS=8
if [ "$RECURSIVE_K" -gt 4 ]; then
  ROLLOUT_STEPS=16
fi

RUN_NAME="grid5_recursive_mlp_${PREDICTOR}_delta_z128_k${RECURSIVE_K}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/${RUN_NAME}"

python -m puzzle_jepa.train.grid5 --config-name grid5_sudoku_sigreg \
  output_dir="${RUN_ROOT}" \
  model.encoder_type=mlp \
  model.predictor_type="${PREDICTOR}" \
  model.predict_delta=true \
  model.latent_size=128 \
  model.encoder_hidden_size=512 \
  model.predictor_hidden_size=512 \
  model.max_rollout_steps="${ROLLOUT_STEPS}" \
  training.rollout_steps="${ROLLOUT_STEPS}" \
  training.recursive_rollout_steps="${RECURSIVE_K}" \
  training.recursive_rollout_weight=1.0

python -m puzzle_jepa.eval.grid5_diagnostics \
  --run-root "${RUN_ROOT}" \
  --output-dir "${RUN_ROOT}/diagnostics" \
  --seed "$((6000 + SLURM_ARRAY_TASK_ID))" \
  --latent-examples 64 \
  --trajectory-examples 32 \
  --rank-examples 32 \
  --detail-examples 4 \
  --plan-examples 16 \
  --plan-beam-size 4 \
  --plan-branch-size 8 \
  --max-plan-steps 96

python -m puzzle_jepa.eval.grid5_mpc_cem_diagnostics \
  --run-root "${RUN_ROOT}" \
  --output-dir "${RUN_ROOT}/diagnostics_mpc_cem" \
  --seed "$((7000 + SLURM_ARRAY_TASK_ID))" \
  --horizons 4 8 16 32 64 \
  --plan-examples 8 \
  --max-mpc-steps 64 \
  --candidates 300 \
  --elites 30 \
  --iterations 10 \
  --qualitative-examples 4 \
  --qualitative-candidates 6 \
  --qualitative-cem-candidates 128 \
  --qualitative-cem-iterations 5
