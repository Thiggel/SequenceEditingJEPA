#!/usr/bin/env bash
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

COUNT="${OLDLOCAL_FAST_COUNT:-18}"
TRAIN_CONCURRENCY="${TRAIN_CONCURRENCY:-12}"
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-12}"
DEPENDENCY_KIND="${EVAL_DEPENDENCY_KIND:-aftercorr}"

if (( COUNT <= 0 )); then
  echo "OLDLOCAL_FAST_COUNT must be positive." >&2
  exit 2
fi

last=$((COUNT - 1))
train_job="$(
  sbatch --parsable \
    --array="0-${last}%${TRAIN_CONCURRENCY}" \
    scripts/slurm/run_grid_goal_oldlocal_fast_train.slurm
)"
eval_job="$(
  sbatch --parsable \
    --array="0-${last}%${EVAL_CONCURRENCY}" \
    --dependency="${DEPENDENCY_KIND}:${train_job}" \
    scripts/slurm/run_grid_goal_oldlocal_fast_eval.slurm
)"

echo "sweep=oldlocal_fast"
echo "train=${train_job}"
echo "eval=${eval_job}"
