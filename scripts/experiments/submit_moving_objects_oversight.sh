#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs
MANIFEST="${MOTION_MANIFEST:?MOTION_MANIFEST is required}"
COUNT="${COUNT:-20}"
CADENCE_HOURS="${CADENCE_HOURS:-6}"

for index in $(seq 0 "$((COUNT - 1))"); do
  begin="now+$((index * CADENCE_HOURS))hours"
  job_id="$(sbatch --parsable --begin="${begin}" --export=ALL,MOTION_MANIFEST="${MANIFEST}" scripts/slurm/run_moving_objects_oversight.slurm)"
  printf 'watcher[%02d]\tbegin=%s\tjob=%s\n' "${index}" "${begin}" "${job_id}"
done
