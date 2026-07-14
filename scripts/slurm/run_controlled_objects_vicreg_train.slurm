#!/usr/bin/env bash
#SBATCH --job-name=co_vicreg_train
#SBATCH --partition=a40,rtxpro6k,a100,a100mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
unset VIRTUAL_ENV
export PATH="/usr/local/bin:/usr/bin:/bin"

TASK_MANIFEST="${TASK_MANIFEST:?TASK_MANIFEST is required}"
STAGE="${STAGE:?STAGE is required}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
ROW="$(awk -F '\t' -v task_id="${TASK_ID}" 'NR > 1 && $1 == task_id {print; exit}' "${TASK_MANIFEST}")"
if [[ -z "${ROW}" ]]; then
  printf 'No task %s in %s.\n' "${TASK_ID}" "${TASK_MANIFEST}" >&2
  exit 1
fi
IFS=$'\t' read -r _ variance_weight covariance_weight seed <<<"${ROW}"
variance_tag="${variance_weight//./p}"
covariance_tag="${covariance_weight//./p}"
common="v${variance_tag}_c${covariance_tag}_s${seed}"

case "${STAGE}" in
  h1)
    run_name="${common}_h1"
    level_spans='[1]'
    train_from_level=0
    init_checkpoint=''
    ;;
  h110)
    run_name="${common}_h110"
    level_spans='[1,10]'
    train_from_level=1
    init_checkpoint="${OUTPUT_ROOT}/${common}_h1/checkpoint.pt"
    ;;
  h110100)
    run_name="${common}_h110100"
    level_spans='[1,10,100]'
    train_from_level=2
    init_checkpoint="${OUTPUT_ROOT}/${common}_h110/checkpoint.pt"
    ;;
  *)
    printf 'Unknown hierarchy stage: %s\n' "${STAGE}" >&2
    exit 1
    ;;
esac

RUN_NAME="${run_name}" \
SEED="${seed}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
MODEL_CONFIG=mlp_hwm_1_10_100 \
OBJECTIVE_CONFIG=ema_vicreg_sweep \
LEVEL_SPANS="${level_spans}" \
PREDICTOR_ARCHITECTURE=transformer \
ROLLOUT_STEPS=4 \
ROLLOUT_ALL_LEVELS=true \
ROLLOUT_LAMBDA=1.0 \
TRAIN_FROM_LEVEL="${train_from_level}" \
INIT_CHECKPOINT="${init_checkpoint}" \
HIDDEN_DIM=256 \
OBJECT_COUNT=2 \
TRAJECTORY_LENGTH=512 \
VICREG_VARIANCE_WEIGHT="${variance_weight}" \
VICREG_COVARIANCE_WEIGHT="${covariance_weight}" \
LDAD_WEIGHT=0 \
MAX_STEPS="${MAX_STEPS:-20000}" \
BATCH_SIZE="${BATCH_SIZE:-64}" \
TRAIN_TRAJECTORIES="${TRAIN_TRAJECTORIES:-512}" \
EVAL_TRAJECTORIES="${EVAL_TRAJECTORIES:-128}" \
PLANNING_EPISODES=0 \
PLANNING_CANDIDATES=512 \
bash scripts/slurm/run_controlled_objects_train.slurm
