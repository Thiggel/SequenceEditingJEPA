#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
unset VIRTUAL_ENV
export PATH="/usr/local/bin:/usr/bin:/bin"
mkdir -p logs
if [[ -z "${PUZZLE_JEPA_WORK_ROOT:-}" ]]; then
  source scripts/env.sh
fi

SWEEP_NAME="${SWEEP_NAME:-controlled_valid_hwm_vicreg_v1_steps${MAX_STEPS:-20000}}"
OUTPUT_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/controlled_objects/${SWEEP_NAME}"
MANIFEST_DIR="${PUZZLE_JEPA_WORK_ROOT}/runs/controlled_objects/manifests"
TASK_MANIFEST="${MANIFEST_DIR}/${SWEEP_NAME}_tasks.tsv"
JOB_MANIFEST="${MANIFEST_DIR}/${SWEEP_NAME}_array_jobs.tsv"
mkdir -p "${MANIFEST_DIR}" "${OUTPUT_ROOT}"

printf 'task_id\tvariance_weight\tcovariance_weight\tseed\n' > "${TASK_MANIFEST}"
task_id=0
for variance_weight in 0.05 1 10 29.409; do
  for covariance_weight in 0.1 1 10 17.866; do
    for seed in 1707 2707 3707; do
      printf '%s\t%s\t%s\t%s\n' \
        "${task_id}" "${variance_weight}" "${covariance_weight}" "${seed}" \
        >> "${TASK_MANIFEST}"
      task_id=$((task_id + 1))
    done
  done
done
if [[ "${task_id}" -ne 48 ]]; then
  printf 'Internal error: expected 48 cells, built %s.\n' "${task_id}" >&2
  exit 1
fi

printf 'stage\tjob_id\tdependency\ttasks\toutput_root\n' > "${JOB_MANIFEST}"
if [[ "${SUBMIT:-0}" != 1 ]]; then
  printf 'DRY RUN: 3 staged train arrays (144 tasks) and 1 final eval array (48 tasks).\n'
  printf 'Final comparison cells: 48. Task manifest: %s\n' "${TASK_MANIFEST}"
  exit 0
fi

array="0-47%${MAX_CONCURRENT:-16}"
exports="ALL,TASK_MANIFEST=${TASK_MANIFEST},OUTPUT_ROOT=${OUTPUT_ROOT},MAX_STEPS=${MAX_STEPS:-20000},BATCH_SIZE=${BATCH_SIZE:-64},TRAIN_TRAJECTORIES=${TRAIN_TRAJECTORIES:-512},EVAL_TRAJECTORIES=${EVAL_TRAJECTORIES:-128}"
submit_train() {
  local stage="$1"
  local dependency="$2"
  local dependency_args=()
  [[ -n "${dependency}" ]] && dependency_args=("--dependency=aftercorr:${dependency}")
  sbatch --parsable --job-name="co_vr_${stage}" --array="${array}" \
    "${dependency_args[@]}" --export="${exports},STAGE=${stage}" \
    scripts/slurm/run_controlled_objects_vicreg_train.slurm
}

h1_job="$(submit_train h1 '')"
h110_job="$(submit_train h110 "${h1_job}")"
h110100_job="$(submit_train h110100 "${h110_job}")"
eval_job="$(sbatch --parsable --job-name=co_vr_eval --array="${array}" \
  --dependency="aftercorr:${h110100_job}" \
  --export="${exports},PROBE_TRAIN_SAMPLES=${PROBE_TRAIN_SAMPLES:-1024},PROBE_EVAL_SAMPLES=${PROBE_EVAL_SAMPLES:-512},PROBE_STEPS=${PROBE_STEPS:-200},PLANNING_EPISODES=${PLANNING_EPISODES:-1},PLANNING_CANDIDATES=${PLANNING_CANDIDATES:-512}" \
  scripts/slurm/run_controlled_objects_vicreg_eval.slurm)"

printf 'h1\t%s\t\t48\t%s\n' "${h1_job}" "${OUTPUT_ROOT}" >> "${JOB_MANIFEST}"
printf 'h110\t%s\t%s\t48\t%s\n' "${h110_job}" "${h1_job}" "${OUTPUT_ROOT}" >> "${JOB_MANIFEST}"
printf 'h110100\t%s\t%s\t48\t%s\n' "${h110100_job}" "${h110_job}" "${OUTPUT_ROOT}" >> "${JOB_MANIFEST}"
printf 'eval\t%s\t%s\t48\t%s\n' "${eval_job}" "${h110100_job}" "${OUTPUT_ROOT}" >> "${JOB_MANIFEST}"

printf 'Submitted fixed HWM VICReg sweep.\n'
printf 'Jobs: h1=%s h110=%s h110100=%s eval=%s\n' \
  "${h1_job}" "${h110_job}" "${h110100_job}" "${eval_job}"
printf 'Task manifest: %s\nJob manifest: %s\nOutput root: %s\n' \
  "${TASK_MANIFEST}" "${JOB_MANIFEST}" "${OUTPUT_ROOT}"
