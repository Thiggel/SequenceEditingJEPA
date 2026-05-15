#!/usr/bin/env bash
#SBATCH --job-name=seqedit_igsm_full_eval
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:59:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/home/hpc/c107fa/c107fa12/sequence-editing}"
source scripts/env.sh

python -m seq_edit_jepa.eval.posthoc_igsm_ood \
  --runs-root "${SEQ_EDIT_JEPA_WORK_ROOT}/runs" \
  --runs \
    igsm_official_med_mask_action_conditioned_jepa_200k \
    igsm_official_med_mask_denoising_lm_200k \
    igsm_official_med_causal_lm_200k \
  --checkpoint-glob "checkpoint-*" \
  --seq-len 1024 \
  --batches 16 \
  --batch-size 8 \
  --ood-op-values 20 21 22 23 \
  --by-op \
  --include-id \
  --full-denoise \
  --print-samples \
  --output "${SEQ_EDIT_JEPA_WORK_ROOT}/posthoc/igsm_ood/full_denoise_metrics_${SLURM_JOB_ID}.jsonl"
