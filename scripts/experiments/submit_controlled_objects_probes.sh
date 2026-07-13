#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs
source scripts/env.sh

SWEEP_NAME="${SWEEP_NAME:-controlled_hierarchy_rollout_v5_steps20000}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-${PUZZLE_JEPA_WORK_ROOT}/runs/controlled_objects/manifests/${SWEEP_NAME}.tsv}"
PROBE_MANIFEST="${PROBE_MANIFEST:-${PUZZLE_JEPA_WORK_ROOT}/runs/controlled_objects/manifests/${SWEEP_NAME}_probes_v2.tsv}"
EXPECTED_JOBS="${EXPECTED_JOBS:-54}"
mkdir -p "$(dirname "${PROBE_MANIFEST}")"
printf 'run_name\tprobe_job_id\ttrain_job_id\tcheckpoint\toutput\n' > "${PROBE_MANIFEST}"

JOB_COUNT=0
while IFS='|' read -r run_name train_job_id output_dir; do
  checkpoint="${output_dir}/checkpoint.pt"
  output="${output_dir}/probe_eval_v2.json"
  if [[ "${SUBMIT:-0}" == "1" ]]; then
    dependency_args=()
    if [[ ! -f "${checkpoint}" ]]; then
      dependency_args=("--dependency=afterok:${train_job_id}")
    fi
    probe_job_id="$(
      sbatch --parsable "${dependency_args[@]}" \
        --export="ALL,CHECKPOINT=${checkpoint},OUTPUT_PATH=${output},PROBE_SEED=${PROBE_SEED:-9917},PROBE_TRAIN_SAMPLES=${PROBE_TRAIN_SAMPLES:-1024},PROBE_EVAL_SAMPLES=${PROBE_EVAL_SAMPLES:-512},PROBE_BATCH_SIZE=${PROBE_BATCH_SIZE:-64},PROBE_STEPS=${PROBE_STEPS:-200},PROBE_LEARNING_RATE=${PROBE_LEARNING_RATE:-0.003}" \
        scripts/slurm/run_controlled_objects_probes.slurm
    )"
  else
    probe_job_id="dry${JOB_COUNT}"
    printf 'DRY RUN: probe=%s train_job=%s\n' "${run_name}" "${train_job_id}"
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${run_name}" "${probe_job_id}" "${train_job_id}" "${checkpoint}" "${output}" \
    >> "${PROBE_MANIFEST}"
  JOB_COUNT=$((JOB_COUNT + 1))
done < <(awk -F '\t' 'NR > 1 {print $1 "|" $2 "|" $16}' "${TRAIN_MANIFEST}")

if [[ "${JOB_COUNT}" -ne "${EXPECTED_JOBS}" ]]; then
  printf 'Expected %s training rows, found %s in %s.\n' \
    "${EXPECTED_JOBS}" "${JOB_COUNT}" "${TRAIN_MANIFEST}" >&2
  exit 1
fi

if [[ "${SUBMIT:-0}" == "1" ]]; then
  printf 'Submitted %s frozen-probe jobs. Manifest: %s\n' \
    "${JOB_COUNT}" "${PROBE_MANIFEST}"
else
  printf 'Dry run only: %s frozen-probe jobs. Re-run with SUBMIT=1 to submit. Manifest: %s\n' \
    "${JOB_COUNT}" "${PROBE_MANIFEST}"
fi
