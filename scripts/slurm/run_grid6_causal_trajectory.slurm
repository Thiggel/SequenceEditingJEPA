#!/bin/bash
#SBATCH --job-name=puzzle_grid6_traj
#SBATCH --output=logs/puzzle_grid6_traj_%A_%a.out
#SBATCH --error=logs/puzzle_grid6_traj_%A_%a.err
#SBATCH --array=0-1%2
#SBATCH --partition=a40,a100,rtxpro6k
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=18:00:00

set -euo pipefail

cd /home/hpc/c107fa/c107fa12/sequence-editing
source ./scripts/env.sh
mkdir -p logs

# name horizons seed
SPECS=(
  "k1 [1] 13001"
  "mh [1,2,4,8,16] 13002"
)

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if (( INDEX < 0 || INDEX >= ${#SPECS[@]} )); then
  echo "Unknown SLURM_ARRAY_TASK_ID=${INDEX}; valid range is 0-$(( ${#SPECS[@]} - 1 ))" >&2
  exit 2
fi

read -r NAME HORIZONS SEED <<<"${SPECS[$INDEX]}"
RUN_NAME="grid6_causal_traj_${NAME}_d320"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/${RUN_NAME}"

python -m puzzle_jepa.train.grid6 --config-name grid6_sudoku_trajectory \
  seed="${SEED}" \
  output_dir="${RUN_ROOT}" \
  model.horizons="${HORIZONS}" \
  model.d_model=320 \
  model.board_hidden_size=1280 \
  model.action_dim=32 \
  model.encoder_layers=4 \
  model.predictor_layers=4 \
  model.action_chunk_layers=2 \
  model.num_heads=8 \
  model.intermediate_size=1280 \
  model.max_sequence_steps=33 \
  model.max_horizon=16 \
  model.stabilizer_type=sigreg \
  model.sigreg_weight=1.0 \
  model.target_encoder_momentum=0.99 \
  training.rollout_steps=32 \
  training.max_steps=5000 \
  training.batch_size=192 \
  training.goal_energy_weight=1.0 \
  training.eval_every_steps=1000 \
  training.save_every_steps=1000 \
  eval.batch_size=64
