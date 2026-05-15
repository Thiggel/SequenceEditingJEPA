#!/usr/bin/env bash
#SBATCH --job-name=seqedit_igsm_step_k_eval
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:59:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/home/hpc/c107fa/c107fa12/sequence-editing}"
mkdir -p logs
source scripts/env.sh

COMMITS=(schedule 1 2 5 10)
COMMIT="${COMMITS[$SLURM_ARRAY_TASK_ID]}"

python -m seq_edit_jepa.eval.posthoc_igsm_ood \
  --runs-root "${SEQ_EDIT_JEPA_WORK_ROOT}/runs" \
  --runs \
    igsm_official_med_step_mask_jepa_T20_200k \
    igsm_official_med_step_mask_jepa_T50_200k \
    igsm_official_med_step_mask_jepa_T64_200k \
  --checkpoint-glob "checkpoint-*" \
  --latest-only \
  --seq-len 1024 \
  --batches 16 \
  --batch-size 8 \
  --ood-op-values 20 21 22 23 \
  --by-op \
  --include-id \
  --full-denoise \
  --commit-k "${COMMIT}" \
  --jepa-inference-mode predictor_decoder \
  --print-samples \
  --output "${SEQ_EDIT_JEPA_WORK_ROOT}/posthoc/igsm_ood/stepwise_commit_k${COMMIT}_${SLURM_JOB_ID}.jsonl"
