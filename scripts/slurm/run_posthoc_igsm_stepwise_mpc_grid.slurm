#!/usr/bin/env bash
#SBATCH --job-name=seqedit_igsm_step_mpc_grid
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:59:00
#SBATCH --array=0-15%4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/home/hpc/c107fa/c107fa12/sequence-editing}"
mkdir -p logs
source scripts/env.sh

HORIZONS=(1 2 4 1 2 4 1 2 4 1 2 4 2 4 2 4)
POSITIONS=(4 4 4 8 8 8 16 16 16 4 4 4 8 8 16 16)
TOKENS=(1 1 1 5 5 5 5 5 5 1 1 1 5 5 5 5)
ROLLOUTS=(latent latent latent latent latent latent latent latent latent reencode reencode reencode reencode reencode reencode reencode)

H="${HORIZONS[$SLURM_ARRAY_TASK_ID]}"
M="${POSITIONS[$SLURM_ARRAY_TASK_ID]}"
K="${TOKENS[$SLURM_ARRAY_TASK_ID]}"
ROLLOUT="${ROLLOUTS[$SLURM_ARRAY_TASK_ID]}"
LABEL="h${H}_m${M}_k${K}_${ROLLOUT}"

python -m seq_edit_jepa.eval.oracle_mpc \
  --runs-root "${SEQ_EDIT_JEPA_WORK_ROOT}/runs" \
  --runs \
    igsm_official_med_step_mask_jepa_T20_200k \
    igsm_official_med_step_mask_jepa_T50_200k \
  --checkpoint-glob "checkpoint-*" \
  --latest-only \
  --seq-len 1024 \
  --batches 4 \
  --batch-size 1 \
  --ood-op-values 20 23 \
  --by-op \
  --include-id \
  --horizon "${H}" \
  --candidates-per-step "${M}" \
  --tokens-per-position "${K}" \
  --max-steps 220 \
  --policy-weight 0.0 \
  --score-mask editable \
  --rollout-mode "${ROLLOUT}" \
  --output "${SEQ_EDIT_JEPA_WORK_ROOT}/posthoc/igsm_ood/stepwise_mpc_grid_${LABEL}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.jsonl"
