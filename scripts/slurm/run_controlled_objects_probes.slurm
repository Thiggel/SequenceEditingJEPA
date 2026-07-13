#!/usr/bin/env bash
#SBATCH --job-name=controlled_probe
#SBATCH --partition=a40,rtxpro6k,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source scripts/env.sh

CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required}"
OUTPUT_PATH="${OUTPUT_PATH:?OUTPUT_PATH is required}"

python -m puzzle_jepa.train.controlled_objects_probes \
  "${CHECKPOINT}" \
  --output "${OUTPUT_PATH}" \
  --device auto \
  --probe-seed "${PROBE_SEED:-9917}" \
  --train-samples "${PROBE_TRAIN_SAMPLES:-1024}" \
  --eval-samples "${PROBE_EVAL_SAMPLES:-512}" \
  --batch-size "${PROBE_BATCH_SIZE:-64}" \
  --steps "${PROBE_STEPS:-200}" \
  --learning-rate "${PROBE_LEARNING_RATE:-0.003}"
