#!/usr/bin/env bash
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

STAGE="${GRID_GOAL_STAGE:-${1:-goal_conditioning}}"
TRAIN_CONCURRENCY="${TRAIN_CONCURRENCY:-4}"
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-4}"
DEPENDENCY_KIND="${EVAL_DEPENDENCY_KIND:-aftercorr}"

case "${STAGE}" in
  goal_conditioning) count=3 ;;
  dense_horizon) count=5 ;;
  hierarchy_levels) count=6 ;;
  predictor_delta_topk) count=4 ;;
  ranking_losses) count=7 ;;
  hierarchical_planning) count=1 ;;
  policy_prior) count=4 ;;
  *)
    echo "Unknown GRID_GOAL_STAGE=${STAGE}" >&2
    exit 2
    ;;
esac

last=$((count - 1))
train_job="$(
  sbatch --parsable \
    --array="0-${last}%${TRAIN_CONCURRENCY}" \
    --export=ALL,GRID_GOAL_STAGE="${STAGE}" \
    scripts/slurm/run_grid_goal_next_train.slurm
)"
eval_job="$(
  sbatch --parsable \
    --array="0-${last}%${EVAL_CONCURRENCY}" \
    --dependency="${DEPENDENCY_KIND}:${train_job}" \
    --export=ALL,GRID_GOAL_STAGE="${STAGE}" \
    scripts/slurm/run_grid_goal_next_eval.slurm
)"

echo "stage=${STAGE}"
echo "train=${train_job}"
echo "eval=${eval_job}"
