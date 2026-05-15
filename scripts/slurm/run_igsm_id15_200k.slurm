#!/usr/bin/env bash
#SBATCH --job-name=seqedit_igsm_id15
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=23:59:00
#SBATCH --array=0-2%1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

CONFIGS=(
  "long/igsm_id15_mask_200k"
  "long/igsm_id15_denoising_lm_200k"
  "long/igsm_id15_causal_lm_200k"
)

python -m seq_edit_jepa.train.hydra_train --config-name "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
