#!/usr/bin/env bash
#SBATCH --job-name=controlled_grid
#SBATCH --partition=a40,rtxpro6k,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

TASK_MANIFEST="${TASK_MANIFEST:?TASK_MANIFEST is required}"
STAGE="${STAGE:?STAGE is required}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
ROW="$(awk -F '\t' -v task_id="${TASK_ID}" 'NR > 1 && $1 == task_id {print; exit}' "${TASK_MANIFEST}")"
if [[ -z "${ROW}" ]]; then
  printf 'No task %s in %s.\n' "${TASK_ID}" "${TASK_MANIFEST}" >&2
  exit 1
fi
IFS=$'\t' read -r _ architecture rollout weighting lambda object_count seed <<<"${ROW}"
common="a${architecture}_r${rollout}_w${weighting}_n${object_count}_s${seed}"

case "${STAGE}" in
  h1)
    run_name="${common}_h1"
    level_spans='[1]'
    train_from_level=0
    init_checkpoint=''
    ;;
  h14)
    run_name="${common}_h14"
    level_spans='[1,4]'
    train_from_level=1
    init_checkpoint="${OUTPUT_ROOT}/${common}_h1/checkpoint.pt"
    ;;
  h1416)
    run_name="${common}_h1416"
    level_spans='[1,4,16]'
    train_from_level=2
    init_checkpoint="${OUTPUT_ROOT}/${common}_h14/checkpoint.pt"
    ;;
  h12)
    run_name="${common}_h12"
    level_spans='[1,2]'
    train_from_level=1
    init_checkpoint="${OUTPUT_ROOT}/${common}_h1/checkpoint.pt"
    ;;
  h124)
    run_name="${common}_h124"
    level_spans='[1,2,4]'
    train_from_level=2
    init_checkpoint="${OUTPUT_ROOT}/${common}_h12/checkpoint.pt"
    ;;
  *)
    printf 'Unknown hierarchy stage: %s\n' "${STAGE}" >&2
    exit 1
    ;;
esac

RUN_NAME="${run_name}" \
SEED="${seed}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
MODEL_CONFIG=mlp_causal_hwm \
OBJECTIVE_CONFIG=ema_vicreg \
LEVEL_SPANS="${level_spans}" \
PREDICTOR_ARCHITECTURE="${architecture}" \
ROLLOUT_STEPS="${rollout}" \
ROLLOUT_ALL_LEVELS=true \
ROLLOUT_LAMBDA="${lambda}" \
TRAIN_FROM_LEVEL="${train_from_level}" \
INIT_CHECKPOINT="${init_checkpoint}" \
HIDDEN_DIM="${HIDDEN_DIM:-256}" \
OBJECT_COUNT="${object_count}" \
LDAD_WEIGHT=0 \
MAX_STEPS="${MAX_STEPS:-20000}" \
BATCH_SIZE="${BATCH_SIZE:-64}" \
PLANNING_EPISODES="$([[ "${STAGE}" == h12 ]] && printf 0 || printf '%s' "${PLANNING_EPISODES:-4}")" \
PLANNING_CANDIDATES="${PLANNING_CANDIDATES:-32}" \
bash scripts/slurm/run_controlled_objects_train.slurm
