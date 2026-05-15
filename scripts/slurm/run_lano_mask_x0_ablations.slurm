#!/usr/bin/env bash
#SBATCH --job-name=seqedit_lano_x0_ablate
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=23:59:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

CONFIGS=(
  "ablations/lano_mask_x0_denoising_lm_20k"
  "ablations/lano_mask_x0_policy_only_20k"
  "ablations/lano_mask_x0_action_free_20k"
  "lano_mask_x0"
)

python -m seq_edit_jepa.train.hydra_train --config-name "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
