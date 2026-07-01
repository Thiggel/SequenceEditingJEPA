#!/usr/bin/env bash
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

COUNT="${MINAUX_WAVE_COUNT:-29}"
TRAIN_CONCURRENCY="${TRAIN_CONCURRENCY:-29}"
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-29}"
DEPENDENCY_KIND="${EVAL_DEPENDENCY_KIND:-aftercorr}"

if (( COUNT <= 0 )); then
  echo "MINAUX_WAVE_COUNT must be positive." >&2
  exit 2
fi

last=$((COUNT - 1))
train_job="$(
  sbatch --parsable \
    --array="0-${last}%${TRAIN_CONCURRENCY}" \
    scripts/slurm/run_grid_goal_minaux_wave_train.slurm
)"
eval_job="$(
  sbatch --parsable \
    --array="0-${last}%${EVAL_CONCURRENCY}" \
    --dependency="${DEPENDENCY_KIND}:${train_job}" \
    scripts/slurm/run_grid_goal_minaux_wave_eval.slurm
)"

echo "sweep=minaux_wave"
echo "train=${train_job}"
echo "eval=${eval_job}"
