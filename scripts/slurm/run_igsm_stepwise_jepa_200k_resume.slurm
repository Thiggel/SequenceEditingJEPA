#!/usr/bin/env bash
#SBATCH --job-name=seqedit_igsm_step_resume
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=23:59:00
#SBATCH --array=0-3%2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh
export IGSM_REPO_PATH="${SEQ_EDIT_JEPA_WORK_ROOT}/external/iGSM"

RUN_DIRS=(
  "${SEQ_EDIT_JEPA_WORK_ROOT}/runs/igsm_official_med_step_mask_jepa_T20_200k"
  "${SEQ_EDIT_JEPA_WORK_ROOT}/runs/igsm_official_med_step_mask_jepa_T50_200k"
  "${SEQ_EDIT_JEPA_WORK_ROOT}/runs/igsm_official_med_step_mask_jepa_T64_200k"
  "${SEQ_EDIT_JEPA_WORK_ROOT}/runs/igsm_official_med_step_replace_jepa_T64_200k"
)

RUN_DIR="${RUN_DIRS[$SLURM_ARRAY_TASK_ID]}"
LATEST_CHECKPOINT="$(
  find "${RUN_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' \
    -exec test -f '{}/trainer_state.json' ';' \
    -exec test -f '{}/model.safetensors' ';' \
    -printf '%f\n' \
    | sort -t- -k2,2n \
    | tail -n 1
)"

if [[ -z "${LATEST_CHECKPOINT}" ]]; then
  echo "No checkpoint found in ${RUN_DIR}" >&2
  exit 1
fi

RESUME_PATH="${RUN_DIR}/${LATEST_CHECKPOINT}"
echo "Resuming ${RUN_DIR} from ${RESUME_PATH}"

python -m seq_edit_jepa.train.resume_run \
  --run-dir "${RUN_DIR}" \
  --checkpoint "${RESUME_PATH}"
