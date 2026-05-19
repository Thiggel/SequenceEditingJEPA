#!/usr/bin/env bash
#SBATCH --job-name=seqedit_igsm_step_diag
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=11:59:00
#SBATCH --array=0-2%3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/home/hpc/c107fa/c107fa12/sequence-editing}"
mkdir -p logs
source scripts/env.sh

DIAGNOSTICS=(coverage factorized factorized)
LABELS=(coverage factorized dlm_proposer)
EXTRA_ARGS=("" "" "--factor-modes model_pos_dlm_token oracle_pos_dlm_token --proposer-run igsm_official_med_mask_x0_denoising_lm_200k")

DIAG="${DIAGNOSTICS[$SLURM_ARRAY_TASK_ID]}"
LABEL="${LABELS[$SLURM_ARRAY_TASK_ID]}"
# shellcheck disable=SC2206
EXTRA=(${EXTRA_ARGS[$SLURM_ARRAY_TASK_ID]})

python -m seq_edit_jepa.eval.stepwise_diagnostics \
  --runs-root "${SEQ_EDIT_JEPA_WORK_ROOT}/runs" \
  --runs \
    igsm_official_med_step_mask_jepa_T20_200k \
    igsm_official_med_step_mask_jepa_T50_200k \
  --checkpoint-glob "checkpoint-*" \
  --latest-only \
  --seq-len 1024 \
  --batches 8 \
  --batch-size 8 \
  --ood-op-values 20 23 \
  --by-op \
  --include-id \
  --diagnostics "${DIAG}" \
  --coverage-rollout-steps 4 \
  --m-values 1 2 4 8 16 \
  --k-values 1 5 20 100 \
  --max-steps 220 \
  "${EXTRA[@]}" \
  --output "${SEQ_EDIT_JEPA_WORK_ROOT}/posthoc/igsm_ood/stepwise_diagnostics_${LABEL}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.jsonl"
