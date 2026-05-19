#!/usr/bin/env bash
#SBATCH --job-name=seqedit_igsm_obj_mpc
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:59:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/home/hpc/c107fa/c107fa12/sequence-editing}"
mkdir -p logs
source scripts/env.sh

RUNS=(
  "igsm_official_med_step_mask_jepa_T20_lewm_no_dec_200k"
  "igsm_official_med_step_mask_jepa_T20_lewm_value_200k"
)
SCORE_MODES=(
  "oracle_goal"
  "value_head"
)

RUN="${RUNS[$SLURM_ARRAY_TASK_ID]}"
SCORE_MODE="${SCORE_MODES[$SLURM_ARRAY_TASK_ID]}"
LABEL="${RUN}_${SCORE_MODE}"

python -m seq_edit_jepa.eval.oracle_mpc \
  --runs-root "${SEQ_EDIT_JEPA_WORK_ROOT}/runs" \
  --runs "${RUN}" \
  --checkpoint-glob "checkpoint-*" \
  --latest-only \
  --seq-len 1024 \
  --batches 4 \
  --batch-size 1 \
  --ood-op-values 20 23 \
  --by-op \
  --include-id \
  --horizon 2 \
  --candidates-per-step 8 \
  --tokens-per-position 5 \
  --max-steps 220 \
  --policy-weight 0.0 \
  --score-mask editable \
  --score-mode "${SCORE_MODE}" \
  --rollout-mode latent \
  --output "${SEQ_EDIT_JEPA_WORK_ROOT}/posthoc/igsm_ood/objective_mpc_${LABEL}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.jsonl"
