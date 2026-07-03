#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs

VARIANT_COUNT="${VARIANT_COUNT:-30}"
CONCURRENCY="${CONCURRENCY:-12}"

train_job="$(
  sbatch --parsable \
    --array="0-$((VARIANT_COUNT - 1))%${CONCURRENCY}" \
    scripts/slurm/run_grid_goal_weekend_train.slurm
)"

printf 'grid_goal_weekend\ttrain=%s\n' "${train_job}"
for index in $(seq 0 "$((VARIANT_COUNT - 1))"); do
  eval_job="$(
    sbatch --parsable \
      --dependency="afterok:${train_job}_${index}" \
      --export=ALL,VARIANT_INDEX="${index}" \
      scripts/slurm/run_grid_goal_weekend_eval.slurm
  )"
  printf 'grid_goal_weekend_eval\tindex=%s\tjob=%s\tdependency=afterok:%s_%s\n' \
    "${index}" "${eval_job}" "${train_job}" "${index}"
done
