#!/usr/bin/env bash
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

STAGE="${GRID_GOAL_STAGE:-goal_conditioning}"
START="${OVERSIGHT_START:-now}"
COUNT="${OVERSIGHT_COUNT:-10}"
INTERVAL_HOURS="${OVERSIGHT_INTERVAL_HOURS:-12}"

for index in $(seq 0 $((COUNT - 1))); do
  if [[ "${START}" == "now" ]]; then
    if (( index == 0 )); then
      begin="now"
    else
      begin="now+$((index * INTERVAL_HOURS))hours"
    fi
  else
    begin="${START}+$((index * INTERVAL_HOURS))hours"
  fi
  sbatch --begin="${begin}" --export=ALL,GRID_GOAL_STAGE="${STAGE}" scripts/slurm/run_grid_goal_oversight.slurm
done
