#!/usr/bin/env bash
#SBATCH --job-name=seqedit_lm1b_mask
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=23:59:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh
python -m seq_edit_jepa.train.hydra_train --config-name lm1b_mask
