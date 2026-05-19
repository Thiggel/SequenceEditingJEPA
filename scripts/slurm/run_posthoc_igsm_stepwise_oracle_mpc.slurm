#!/usr/bin/env bash
#SBATCH --job-name=seqedit_igsm_step_oracle_mpc
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:59:00
#SBATCH --array=0-5%3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/home/hpc/c107fa/c107fa12/sequence-editing}"
mkdir -p logs
source scripts/env.sh

HORIZONS=(1 2 4 1 2 4)
ORACLE_TOKEN=(0 0 0 1 1 1)
HORIZON="${HORIZONS[$SLURM_ARRAY_TASK_ID]}"
USE_ORACLE_TOKEN="${ORACLE_TOKEN[$SLURM_ARRAY_TASK_ID]}"
LABEL="h${HORIZON}_oracleTok${USE_ORACLE_TOKEN}"

ORACLE_TOKEN_FLAG=()
if [[ "${USE_ORACLE_TOKEN}" == "1" ]]; then
  ORACLE_TOKEN_FLAG=(--include-oracle-token)
fi

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
  --horizon "${HORIZON}" \
  --candidates-per-step 4 \
  --tokens-per-position 1 \
  --max-steps 160 \
  --policy-weight 0.0 \
  --score-mask editable \
  "${ORACLE_TOKEN_FLAG[@]}" \
  --output "${SEQ_EDIT_JEPA_WORK_ROOT}/posthoc/igsm_ood/stepwise_oracle_mpc_${LABEL}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.jsonl"
