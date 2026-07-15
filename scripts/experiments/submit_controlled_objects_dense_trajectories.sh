#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
unset VIRTUAL_ENV
export PATH="/usr/local/bin:/usr/bin:/bin"
mkdir -p logs
if [[ -z "${PUZZLE_JEPA_WORK_ROOT:-}" ]]; then
  source scripts/env.sh
fi

SWEEP_NAME="${SWEEP_NAME:-controlled_dense_trajectories_v1}"
OUTPUT_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/controlled_objects/${SWEEP_NAME}"
MANIFEST_DIR="${PUZZLE_JEPA_WORK_ROOT}/runs/controlled_objects/manifests"
TASK_MANIFEST="${MANIFEST_DIR}/${SWEEP_NAME}_tasks.tsv"
JOB_MANIFEST="${MANIFEST_DIR}/${SWEEP_NAME}_jobs.tsv"
mkdir -p "${MANIFEST_DIR}" "${OUTPUT_ROOT}"

printf 'task_id\tobjective\tsetting\ttrajectory_length\tbatch_size\tmax_steps\twarmup_steps\tseed\n' > "${TASK_MANIFEST}"
task_id=0
append_setting() {
  local objective="$1" setting="$2" trajectory_length="$3" batch_size="$4"
  local max_steps="$5" warmup_steps="$6"
  for seed in 1707 2707 3707; do
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${task_id}" "${objective}" "${setting}" "${trajectory_length}" \
      "${batch_size}" "${max_steps}" "${warmup_steps}" "${seed}" >> "${TASK_MANIFEST}"
    task_id=$((task_id + 1))
  done
}

for objective in vicreg ema_vicreg; do
  append_setting "${objective}" t100_b64 100 64 20000 500
  append_setting "${objective}" t300_b20 300 20 20000 500
  append_setting "${objective}" t500_b12 500 12 20000 500
  append_setting "${objective}" t300_b8 300 8 50000 1250
  append_setting "${objective}" t300_b16 300 16 25000 625
  append_setting "${objective}" t300_b32 300 32 12500 313
  append_setting "${objective}" t300_b64 300 64 6250 156
done

if [[ "${task_id}" -ne 42 ]]; then
  printf 'Internal error: expected 42 tasks, built %s.\n' "${task_id}" >&2
  exit 1
fi

printf 'stage\tjob_id\tdependency\ttasks\toutput_root\n' > "${JOB_MANIFEST}"
if [[ "${SUBMIT:-0}" != 1 ]]; then
  printf 'DRY RUN: 42 dense-trajectory trainers and 42 correlated probes.\n'
  printf 'Task manifest: %s\nOutput root: %s\n' "${TASK_MANIFEST}" "${OUTPUT_ROOT}"
  exit 0
fi

array="0-41%${MAX_CONCURRENT:-12}"
exports="ALL,TASK_MANIFEST=${TASK_MANIFEST},OUTPUT_ROOT=${OUTPUT_ROOT},TRAIN_TRAJECTORIES=${TRAIN_TRAJECTORIES:-512},EVAL_TRAJECTORIES=${EVAL_TRAJECTORIES:-128}"
train_job="$(sbatch --parsable --job-name=co_dense_traj --array="${array}" \
  --export="${exports}" scripts/slurm/run_controlled_objects_dense_trajectory_train.slurm)"
probe_job="$(sbatch --parsable --job-name=co_dense_probe --array="${array}" \
  --dependency="aftercorr:${train_job}" \
  --export="${exports},PROBE_TRAIN_SAMPLES=${PROBE_TRAIN_SAMPLES:-1024},PROBE_EVAL_SAMPLES=${PROBE_EVAL_SAMPLES:-512},PROBE_STEPS=${PROBE_STEPS:-500}" \
  scripts/slurm/run_controlled_objects_dense_trajectory_probe.slurm)"

printf 'train\t%s\t\t42\t%s\n' "${train_job}" "${OUTPUT_ROOT}" >> "${JOB_MANIFEST}"
printf 'probe\t%s\t%s\t42\t%s\n' "${probe_job}" "${train_job}" "${OUTPUT_ROOT}" >> "${JOB_MANIFEST}"
printf 'Submitted dense-trajectory gate: train=%s probe=%s\n' "${train_job}" "${probe_job}"
printf 'Task manifest: %s\nJob manifest: %s\nOutput root: %s\n' \
  "${TASK_MANIFEST}" "${JOB_MANIFEST}" "${OUTPUT_ROOT}"
