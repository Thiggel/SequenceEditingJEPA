#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
unset VIRTUAL_ENV
export PATH="/usr/local/bin:/usr/bin:/bin"
mkdir -p logs
if [[ -z "${PUZZLE_JEPA_WORK_ROOT:-}" ]]; then
  source scripts/env.sh
fi

SWEEP_NAME="${SWEEP_NAME:-controlled_objective_weights_v1_steps${MAX_STEPS:-20000}}"
OUTPUT_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/controlled_objects/${SWEEP_NAME}"
MANIFEST_DIR="${PUZZLE_JEPA_WORK_ROOT}/runs/controlled_objects/manifests"
TASK_MANIFEST="${MANIFEST_DIR}/${SWEEP_NAME}_tasks.tsv"
JOB_MANIFEST="${MANIFEST_DIR}/${SWEEP_NAME}_jobs.tsv"
mkdir -p "${MANIFEST_DIR}" "${OUTPUT_ROOT}"

printf 'task_id\tvariant\ttarget_contract\tstabilizer\tldad_multiplier\tstabilizer_multiplier\tseed\n' > "${TASK_MANIFEST}"
task_id=0
append_task() {
  local variant="$1" target_contract="$2" stabilizer="$3"
  local ldad_multiplier="$4" stabilizer_multiplier="$5"
  for seed in 1707 2707 3707; do
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${task_id}" "${variant}" "${target_contract}" "${stabilizer}" \
      "${ldad_multiplier}" "${stabilizer_multiplier}" "${seed}" >> "${TASK_MANIFEST}"
    task_id=$((task_id + 1))
  done
}

append_task stopgrad stopgrad none 0 0
append_task ema ema none 0 0

for target_contract in online stopgrad ema; do
  for ldad_multiplier in 1 10 100; do
    append_task "ldad_${target_contract}_l${ldad_multiplier}" \
      "${target_contract}" none "${ldad_multiplier}" 0
  done
done

for stabilizer in vicreg sigreg; do
  for target_contract in stopgrad ema; do
    for stabilizer_multiplier in 1 10 100; do
      append_task "${stabilizer}_${target_contract}_l${stabilizer_multiplier}" \
        "${target_contract}" "${stabilizer}" 0 "${stabilizer_multiplier}"
    done
  done
  for target_contract in online stopgrad ema; do
    for ldad_multiplier in 1 10 100; do
      for stabilizer_multiplier in 1 10 100; do
        append_task "ldad_${stabilizer}_${target_contract}_l${ldad_multiplier}_${stabilizer_multiplier}" \
          "${target_contract}" "${stabilizer}" "${ldad_multiplier}" \
          "${stabilizer_multiplier}"
      done
    done
  done
done

if [[ "${task_id}" -ne 231 ]]; then
  printf 'Internal error: expected 231 tasks, built %s.\n' "${task_id}" >&2
  exit 1
fi

printf 'stage\tjob_id\tdependency\ttasks\toutput_root\n' > "${JOB_MANIFEST}"
if [[ "${SUBMIT:-0}" != 1 ]]; then
  printf 'DRY RUN: 231 trainers and 231 correlated probes.\n'
  printf 'Task manifest: %s\nOutput root: %s\n' "${TASK_MANIFEST}" "${OUTPUT_ROOT}"
  exit 0
fi

array="0-230%${MAX_CONCURRENT:-24}"
exports="ALL,TASK_MANIFEST=${TASK_MANIFEST},OUTPUT_ROOT=${OUTPUT_ROOT},MAX_STEPS=${MAX_STEPS:-20000},BATCH_SIZE=${BATCH_SIZE:-64},TRAIN_TRAJECTORIES=${TRAIN_TRAJECTORIES:-512},EVAL_TRAJECTORIES=${EVAL_TRAJECTORIES:-128}"
train_job="$(sbatch --parsable --job-name=co_obj_weight --array="${array}" \
  --export="${exports}" scripts/slurm/run_controlled_objects_objective_weight_train.slurm)"
probe_job="$(sbatch --parsable --job-name=co_obj_probe --array="${array}" \
  --dependency="aftercorr:${train_job}" \
  --export="${exports},PROBE_TRAIN_SAMPLES=${PROBE_TRAIN_SAMPLES:-1024},PROBE_EVAL_SAMPLES=${PROBE_EVAL_SAMPLES:-512},PROBE_STEPS=${PROBE_STEPS:-500}" \
  scripts/slurm/run_controlled_objects_objective_weight_probe.slurm)"

printf 'train\t%s\t\t231\t%s\n' "${train_job}" "${OUTPUT_ROOT}" >> "${JOB_MANIFEST}"
printf 'probe\t%s\t%s\t231\t%s\n' "${probe_job}" "${train_job}" "${OUTPUT_ROOT}" >> "${JOB_MANIFEST}"
printf 'Submitted objective-weight gate: train=%s probe=%s\n' "${train_job}" "${probe_job}"
printf 'Task manifest: %s\nJob manifest: %s\nOutput root: %s\n' \
  "${TASK_MANIFEST}" "${JOB_MANIFEST}" "${OUTPUT_ROOT}"
