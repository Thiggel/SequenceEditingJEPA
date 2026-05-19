#!/usr/bin/env bash
#SBATCH --job-name=seqedit_igsm_value_head
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:59:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/home/hpc/c107fa/c107fa12/sequence-editing}"
mkdir -p logs
source scripts/env.sh

RUNS=(
  "igsm_official_med_step_mask_jepa_T20_200k"
  "igsm_official_med_step_mask_jepa_T50_200k"
)

RUN="${RUNS[$SLURM_ARRAY_TASK_ID]}"
RUN_DIR="${SEQ_EDIT_JEPA_WORK_ROOT}/runs/${RUN}"
LATEST_CHECKPOINT="$(
  find "${RUN_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' \
    -exec test -f '{}/config.json' ';' \
    -exec test -f '{}/model.safetensors' ';' \
    -printf '%f\n' \
    | sort -t- -k2,2n \
    | tail -n 1
)"

if [[ -z "${LATEST_CHECKPOINT}" ]]; then
  echo "No checkpoint found in ${RUN_DIR}" >&2
  exit 1
fi

python -m seq_edit_jepa.train.train_value_head \
  --run-dir "${RUN_DIR}" \
  --checkpoint "${RUN_DIR}/${LATEST_CHECKPOINT}" \
  --output-dir "${SEQ_EDIT_JEPA_WORK_ROOT}/posthoc/value_heads/${RUN}_${LATEST_CHECKPOINT}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
  --seq-len 1024 \
  --steps 5000 \
  --batch-size 16 \
  --lr 0.001 \
  --token-loss-weight 0.25 \
  --log-every 50 \
  --save-every 1000
