#!/usr/bin/env bash
#SBATCH --job-name=motion_probe
#SBATCH --partition=a40,rtxpro6k,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source scripts/env.sh

RUN_NAME="${RUN_NAME:?RUN_NAME is required}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/moving_objects/${RUN_NAME}"
python -m puzzle_jepa.eval.moving_object_probes \
  --checkpoint "${RUN_ROOT}/checkpoint.pt" \
  --output "${RUN_ROOT}/probe_eval_v4.json" \
  --train-samples "${PROBE_TRAIN_SAMPLES:-512}" \
  --eval-samples "${PROBE_EVAL_SAMPLES:-256}" \
  --batch-size "${PROBE_BATCH_SIZE:-64}" \
  --steps "${PROBE_STEPS:-100}"
