#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
MANIFEST="${MOTION_MANIFEST:?MOTION_MANIFEST is required}"
COUNT=0
while IFS=$'\t' read -r run_name job_id latent_dim max_objects seed; do
  if [[ "${run_name}" == "run_name" ]]; then
    continue
  fi
  if [[ "${SUBMIT:-0}" == "1" ]]; then
    dependency_args=()
    if [[ "${DEPEND_ON_TRAIN:-0}" == "1" ]]; then
      dependency_args=(--dependency="afterany:${job_id}")
    fi
    probe_job="$(
      RUN_NAME="${run_name}" \
        sbatch --parsable "${dependency_args[@]}" scripts/slurm/run_moving_objects_probe_eval.slurm
    )"
    printf '%s\ttrain=%s\tprobe=%s\tz=%s\tN=%s\tseed=%s\n' \
      "${run_name}" "${job_id}" "${probe_job}" "${latent_dim}" "${max_objects}" "${seed}"
  else
    printf 'DRY RUN: %s z=%s N=%s seed=%s\n' "${run_name}" "${latent_dim}" "${max_objects}" "${seed}"
  fi
  COUNT=$((COUNT + 1))
done < "${MANIFEST}"
printf 'Rows: %s\n' "${COUNT}"
