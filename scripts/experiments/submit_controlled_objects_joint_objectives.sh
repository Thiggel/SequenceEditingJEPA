#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
unset VIRTUAL_ENV
export PATH="/usr/local/bin:/usr/bin:/bin"
mkdir -p logs
if [[ -z "${PUZZLE_JEPA_WORK_ROOT:-}" ]]; then
  source scripts/env.sh
fi

SWEEP_NAME="${SWEEP_NAME:-controlled_joint_hwm_objectives_v1_steps${MAX_STEPS:-20000}}"
OUTPUT_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/controlled_objects/${SWEEP_NAME}"
MANIFEST_DIR="${PUZZLE_JEPA_WORK_ROOT}/runs/controlled_objects/manifests"
TASK_MANIFEST="${MANIFEST_DIR}/${SWEEP_NAME}_tasks.tsv"
JOB_MANIFEST="${MANIFEST_DIR}/${SWEEP_NAME}_jobs.tsv"
mkdir -p "${MANIFEST_DIR}" "${OUTPUT_ROOT}"

objectives=(
  online
  ema
  sigreg
  ema_sigreg
  vicreg
  ema_vicreg
  ldad
  ema_ldad
  vicreg_ldad
  ema_vicreg_ldad
  sigreg_ldad
  ema_sigreg_ldad
)
printf 'task_id\tobjective\tseed\n' > "${TASK_MANIFEST}"
task_id=0
for objective in "${objectives[@]}"; do
  for seed in 1707 2707 3707; do
    printf '%s\t%s\t%s\n' "${task_id}" "${objective}" "${seed}" >> "${TASK_MANIFEST}"
    task_id=$((task_id + 1))
  done
done
if [[ "${task_id}" -ne 36 ]]; then
  printf 'Internal error: expected 36 cells, built %s.\n' "${task_id}" >&2
  exit 1
fi

printf 'stage\tjob_id\tdependency\ttasks\toutput_root\n' > "${JOB_MANIFEST}"
if [[ "${SUBMIT:-0}" != 1 ]]; then
  printf 'DRY RUN: 36 joint [1,10,100] trainers and 36 dependent probes.\n'
  printf 'Task manifest: %s\nOutput root: %s\n' "${TASK_MANIFEST}" "${OUTPUT_ROOT}"
  exit 0
fi

array="0-35%${MAX_CONCURRENT:-12}"
exports="ALL,TASK_MANIFEST=${TASK_MANIFEST},OUTPUT_ROOT=${OUTPUT_ROOT},MAX_STEPS=${MAX_STEPS:-20000},BATCH_SIZE=${BATCH_SIZE:-64},TRAIN_TRAJECTORIES=${TRAIN_TRAJECTORIES:-512},EVAL_TRAJECTORIES=${EVAL_TRAJECTORIES:-128}"
train_job="$(sbatch --parsable --job-name=co_joint_train --array="${array}" \
  --export="${exports}" scripts/slurm/run_controlled_objects_joint_train.slurm)"
probe_job="$(sbatch --parsable --job-name=co_joint_probe --array="${array}" \
  --dependency="aftercorr:${train_job}" \
  --export="${exports},PROBE_TRAIN_SAMPLES=${PROBE_TRAIN_SAMPLES:-1024},PROBE_EVAL_SAMPLES=${PROBE_EVAL_SAMPLES:-512},PROBE_STEPS=${PROBE_STEPS:-500}" \
  scripts/slurm/run_controlled_objects_joint_probe.slurm)"

printf 'train\t%s\t\t36\t%s\n' "${train_job}" "${OUTPUT_ROOT}" >> "${JOB_MANIFEST}"
printf 'probe\t%s\t%s\t36\t%s\n' "${probe_job}" "${train_job}" "${OUTPUT_ROOT}" >> "${JOB_MANIFEST}"
printf 'Submitted joint HWM objective gate.\nTrain=%s probe=%s\n' "${train_job}" "${probe_job}"
printf 'Task manifest: %s\nJob manifest: %s\nOutput root: %s\n' \
  "${TASK_MANIFEST}" "${JOB_MANIFEST}" "${OUTPUT_ROOT}"
