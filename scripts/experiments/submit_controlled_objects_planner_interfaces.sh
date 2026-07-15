#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
unset VIRTUAL_ENV
export PATH="/usr/local/bin:/usr/bin:/bin"
mkdir -p logs
if [[ -z "${PUZZLE_JEPA_WORK_ROOT:-}" ]]; then
  source scripts/env.sh
fi

SWEEP_NAME="${SWEEP_NAME:-controlled_planner_interfaces_v1}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${PUZZLE_JEPA_WORK_ROOT}/runs/controlled_objects/controlled_joint_hwm_objectives_v1_steps20000}"
OUTPUT_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/controlled_objects/${SWEEP_NAME}"
MANIFEST_DIR="${PUZZLE_JEPA_WORK_ROOT}/runs/controlled_objects/manifests"
TASK_MANIFEST="${MANIFEST_DIR}/${SWEEP_NAME}_tasks.tsv"
JOB_MANIFEST="${MANIFEST_DIR}/${SWEEP_NAME}_jobs.tsv"
mkdir -p "${MANIFEST_DIR}" "${OUTPUT_ROOT}"

printf 'task_id\tobjective\tseed\tmode\thard_project\treachability_weight\n' > "${TASK_MANIFEST}"
task_id=0
append_mode() {
  local objective="$1" seed="$2" mode="$3" hard_project="$4" weight="$5"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${task_id}" "${objective}" "${seed}" "${mode}" "${hard_project}" "${weight}" >> "${TASK_MANIFEST}"
  task_id=$((task_id + 1))
}
for objective in vicreg ema_vicreg; do
  for seed in 1707 2707 3707; do
    append_mode "${objective}" "${seed}" baseline false 0
    append_mode "${objective}" "${seed}" hard_support true 0
    for weight in 0.1 1 10; do
      append_mode "${objective}" "${seed}" "reach_${weight}" false "${weight}"
      append_mode "${objective}" "${seed}" "hard_reach_${weight}" true "${weight}"
    done
  done
done
if [[ "${task_id}" -ne 48 ]]; then
  printf 'Internal error: expected 48 tasks, built %s.\n' "${task_id}" >&2
  exit 1
fi

printf 'stage\tjob_id\tdependency\ttasks\toutput_root\n' > "${JOB_MANIFEST}"
if [[ "${SUBMIT:-0}" != 1 ]]; then
  printf 'DRY RUN: 48 planner-interface evaluations.\n'
  printf 'Task manifest: %s\nOutput root: %s\n' "${TASK_MANIFEST}" "${OUTPUT_ROOT}"
  exit 0
fi

exports="ALL,TASK_MANIFEST=${TASK_MANIFEST},OUTPUT_ROOT=${OUTPUT_ROOT},CHECKPOINT_ROOT=${CHECKPOINT_ROOT},PLANNING_EPISODES=${PLANNING_EPISODES:-32},PLANNING_CANDIDATES=${PLANNING_CANDIDATES:-512}"
job_id="$(sbatch --parsable --job-name=co_plan_iface \
  --array="0-47%${MAX_CONCURRENT:-12}" --export="${exports}" \
  scripts/slurm/run_controlled_objects_planner_interface.slurm)"
printf 'planning\t%s\t\t48\t%s\n' "${job_id}" "${OUTPUT_ROOT}" >> "${JOB_MANIFEST}"
printf 'Submitted planner interfaces: job=%s\n' "${job_id}"
printf 'Task manifest: %s\nJob manifest: %s\nOutput root: %s\n' \
  "${TASK_MANIFEST}" "${JOB_MANIFEST}" "${OUTPUT_ROOT}"
