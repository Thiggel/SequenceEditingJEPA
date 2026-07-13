#!/usr/bin/env bash
#SBATCH --job-name=controlled_probe_grid
#SBATCH --partition=a40,rtxpro6k,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
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
IFS=$'\t' read -r _ architecture rollout weighting _ object_count seed <<<"${ROW}"
common="a${architecture}_r${rollout}_w${weighting}_n${object_count}_s${seed}"
case "${STAGE}" in
  h1|h14|h1416|h124) run_name="${common}_${STAGE}" ;;
  *) printf 'Unknown probe stage: %s\n' "${STAGE}" >&2; exit 1 ;;
esac

CHECKPOINT="${OUTPUT_ROOT}/${run_name}/checkpoint.pt" \
OUTPUT_PATH="${OUTPUT_ROOT}/${run_name}/probe_eval_v3.json" \
PROBE_SEED="${PROBE_SEED:-9917}" \
PROBE_TRAIN_SAMPLES="${PROBE_TRAIN_SAMPLES:-1024}" \
PROBE_EVAL_SAMPLES="${PROBE_EVAL_SAMPLES:-512}" \
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-64}" \
PROBE_STEPS="${PROBE_STEPS:-200}" \
PROBE_LEARNING_RATE="${PROBE_LEARNING_RATE:-0.003}" \
bash scripts/slurm/run_controlled_objects_probes.slurm
