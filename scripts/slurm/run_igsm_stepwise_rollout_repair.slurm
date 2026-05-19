#!/usr/bin/env bash
#SBATCH --job-name=seqedit_igsm_step_rollrepair
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=23:59:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/home/hpc/c107fa/c107fa12/sequence-editing}"
mkdir -p logs
source scripts/env.sh

SOURCE_RUNS=(
  "igsm_official_med_step_mask_jepa_T20_200k"
  "igsm_official_med_step_mask_jepa_T50_200k"
)
N_STEPS=(20 50)
TRACE_STEPS=("20,10,1" "50,25,1")

SOURCE_RUN="${SOURCE_RUNS[$SLURM_ARRAY_TASK_ID]}"
N="${N_STEPS[$SLURM_ARRAY_TASK_ID]}"
TRACE="${TRACE_STEPS[$SLURM_ARRAY_TASK_ID]}"
SOURCE_DIR="${SEQ_EDIT_JEPA_WORK_ROOT}/runs/${SOURCE_RUN}"
LATEST_CHECKPOINT="$(
  find "${SOURCE_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' \
    -exec test -f '{}/config.json' ';' \
    -exec test -f '{}/model.safetensors' ';' \
    -printf '%f\n' \
    | sort -t- -k2,2n \
    | tail -n 1
)"

if [[ -z "${LATEST_CHECKPOINT}" ]]; then
  echo "No checkpoint found in ${SOURCE_DIR}" >&2
  exit 1
fi

INIT_PATH="${SOURCE_DIR}/${LATEST_CHECKPOINT}"
TARGET_NAME="igsm_official_med_step_mask_jepa_T${N}_rollout_repair"
echo "Initializing ${TARGET_NAME} from ${INIT_PATH}"

python -m seq_edit_jepa.train.hydra_train \
  --config-name long/igsm_official_step_mask_jepa_rollout_repair \
  "long.experiment.name=${TARGET_NAME}" \
  "+long.experiment.init_from_checkpoint=${INIT_PATH}" \
  "long.corruptor.num_steps=${N}" \
  "long.eval.sample_generations.trace_steps=[${TRACE}]"
