#!/usr/bin/env bash
#SBATCH --job-name=seqedit_cfg
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=23:59:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

if [[ $# -lt 1 && -z "${SEQ_EDIT_CONFIG:-}" ]]; then
  echo "Usage: sbatch $0 <config.yaml> or set SEQ_EDIT_CONFIG" >&2
  exit 2
fi

CONFIG="${SEQ_EDIT_CONFIG:-$1}"
CONFIG_NAME="${CONFIG#configs/}"
CONFIG_NAME="${CONFIG_NAME%.yaml}"
python -m seq_edit_jepa.train.hydra_train --config-name "$CONFIG_NAME"
