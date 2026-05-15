#!/usr/bin/env bash
#SBATCH --job-name=seqedit_igsm_x0_k_eval
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:59:00
#SBATCH --array=0-5
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/home/hpc/c107fa/c107fa12/sequence-editing}"
mkdir -p logs
source scripts/env.sh

MODES=(policy_head predictor_decoder predictor_decoder predictor_decoder predictor_decoder predictor_decoder)
COMMITS=(schedule schedule 1 2 5 10)
MODE="${MODES[$SLURM_ARRAY_TASK_ID]}"
COMMIT="${COMMITS[$SLURM_ARRAY_TASK_ID]}"
LABEL="${MODE}_k${COMMIT}"

python -m seq_edit_jepa.eval.posthoc_igsm_ood \
  --runs-root "${SEQ_EDIT_JEPA_WORK_ROOT}/runs" \
  --runs \
    igsm_official_med_mask_x0_action_conditioned_jepa_200k \
    igsm_official_med_mask_x0_denoising_lm_200k \
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
  --jepa-inference-mode "${MODE}" \
  --print-samples \
  --output "${SEQ_EDIT_JEPA_WORK_ROOT}/posthoc/igsm_ood/x0_commit_${LABEL}_${SLURM_JOB_ID}.jsonl"
