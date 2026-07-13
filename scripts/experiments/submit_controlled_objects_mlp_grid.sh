#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs
source scripts/env.sh

SWEEP_NAME="${SWEEP_NAME:-controlled_mlp_hwm_v1_steps${MAX_STEPS:-20000}}"
OUTPUT_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/controlled_objects/${SWEEP_NAME}"
MANIFEST_DIR="${PUZZLE_JEPA_WORK_ROOT}/runs/controlled_objects/manifests"
TASK_MANIFEST="${MANIFEST_DIR}/${SWEEP_NAME}_tasks.tsv"
JOB_MANIFEST="${MANIFEST_DIR}/${SWEEP_NAME}_array_jobs.tsv"
mkdir -p "${MANIFEST_DIR}" "${OUTPUT_ROOT}"

printf 'task_id\tarchitecture\trollout_steps\tweighting\tlambda\tobject_count\tseed\n' > "${TASK_MANIFEST}"
task_id=0
for architecture in transformer gated_deltanet lstm; do
  for rollout in 1 2 4 8; do
    for weighting in unweighted weighted; do
      lambda=1.0
      [[ "${weighting}" == weighted ]] && lambda=0.9
      for object_count in 1 2 4 8; do
        for seed in 1707 2707 3707; do
          printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${task_id}" "${architecture}" "${rollout}" "${weighting}" \
            "${lambda}" "${object_count}" "${seed}" >> "${TASK_MANIFEST}"
          task_id=$((task_id + 1))
        done
      done
    done
  done
done
if [[ "${task_id}" -ne 288 ]]; then
  printf 'Internal error: expected 288 base tasks, built %s.\n' "${task_id}" >&2
  exit 1
fi

printf 'stage\tjob_id\tdependency\ttasks\toutput_root\n' > "${JOB_MANIFEST}"
if [[ "${SUBMIT:-0}" != 1 ]]; then
  printf 'DRY RUN: 5 train arrays (1440 tasks) and 4 probe arrays (1152 tasks).\n'
  printf 'Final comparison cells: 1152. Task manifest: %s\n' "${TASK_MANIFEST}"
  exit 0
fi

array="0-287%${MAX_CONCURRENT:-24}"
exports="ALL,TASK_MANIFEST=${TASK_MANIFEST},OUTPUT_ROOT=${OUTPUT_ROOT},MAX_STEPS=${MAX_STEPS:-20000},BATCH_SIZE=${BATCH_SIZE:-64},HIDDEN_DIM=${HIDDEN_DIM:-256},PLANNING_EPISODES=${PLANNING_EPISODES:-4},PLANNING_CANDIDATES=${PLANNING_CANDIDATES:-32}"
submit_train() {
  local stage="$1"
  local dependency="$2"
  local dependency_args=()
  [[ -n "${dependency}" ]] && dependency_args=("--dependency=aftercorr:${dependency}")
  sbatch --parsable --job-name="co_${stage}" --array="${array}" \
    "${dependency_args[@]}" --export="${exports},STAGE=${stage}" \
    scripts/slurm/run_controlled_objects_grid_train.slurm
}
submit_probe() {
  local stage="$1"
  local dependency="$2"
  sbatch --parsable --job-name="co_probe_${stage}" --array="${array}" \
    --dependency="aftercorr:${dependency}" \
    --export="${exports},STAGE=${stage},PROBE_TRAIN_SAMPLES=${PROBE_TRAIN_SAMPLES:-1024},PROBE_EVAL_SAMPLES=${PROBE_EVAL_SAMPLES:-512},PROBE_STEPS=${PROBE_STEPS:-200}" \
    scripts/slurm/run_controlled_objects_grid_probes.slurm
}

h1_job="$(submit_train h1 '')"
h14_job="$(submit_train h14 "${h1_job}")"
h1416_job="$(submit_train h1416 "${h14_job}")"
h12_job="$(submit_train h12 "${h1_job}")"
h124_job="$(submit_train h124 "${h12_job}")"
printf 'h1\t%s\t\t288\t%s\n' "${h1_job}" "${OUTPUT_ROOT}" >> "${JOB_MANIFEST}"
printf 'h14\t%s\t%s\t288\t%s\n' "${h14_job}" "${h1_job}" "${OUTPUT_ROOT}" >> "${JOB_MANIFEST}"
printf 'h1416\t%s\t%s\t288\t%s\n' "${h1416_job}" "${h14_job}" "${OUTPUT_ROOT}" >> "${JOB_MANIFEST}"
printf 'h12\t%s\t%s\t288\t%s\n' "${h12_job}" "${h1_job}" "${OUTPUT_ROOT}" >> "${JOB_MANIFEST}"
printf 'h124\t%s\t%s\t288\t%s\n' "${h124_job}" "${h12_job}" "${OUTPUT_ROOT}" >> "${JOB_MANIFEST}"

for entry in "h1:${h1_job}" "h14:${h14_job}" "h1416:${h1416_job}" "h124:${h124_job}"; do
  stage="${entry%%:*}"
  dependency="${entry#*:}"
  probe_job="$(submit_probe "${stage}" "${dependency}")"
  printf 'probe_%s\t%s\t%s\t288\t%s\n' \
    "${stage}" "${probe_job}" "${dependency}" "${OUTPUT_ROOT}" >> "${JOB_MANIFEST}"
done

printf 'Submitted 5 train arrays and 4 probe arrays for 1152 final cells.\n'
printf 'Task manifest: %s\nJob manifest: %s\nOutput root: %s\n' \
  "${TASK_MANIFEST}" "${JOB_MANIFEST}" "${OUTPUT_ROOT}"
