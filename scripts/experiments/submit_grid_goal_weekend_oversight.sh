#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs

COUNT="${COUNT:-20}"
CADENCE_HOURS="${CADENCE_HOURS:-12}"
RUN_SUFFIX="${RUN_SUFFIX:-${GRID_GOAL_WEEKEND_RUN_SUFFIX:-}}"

for index in $(seq 0 "$((COUNT - 1))"); do
  begin="now+$((index * CADENCE_HOURS))hours"
  job="$(
    sbatch --parsable \
      --begin="${begin}" \
      --export=ALL,GRID_GOAL_WEEKEND_REPAIR_EVALS="${GRID_GOAL_WEEKEND_REPAIR_EVALS:-1}",GRID_GOAL_WEEKEND_RUN_SUFFIX="${RUN_SUFFIX}",RUN_SUFFIX="${RUN_SUFFIX}" \
      scripts/slurm/run_grid_goal_weekend_oversight.slurm
  )"
  printf 'oversight[%02d]\tbegin=%s\tjob=%s\n' "${index}" "${begin}" "${job}"
done
