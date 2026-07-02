#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs

train_job="$(
  sbatch --parsable \
    scripts/slurm/run_grid_goal_dense_exact_train.slurm
)"
eval_job="$(
  sbatch --parsable \
    --dependency="aftercorr:${train_job}" \
    scripts/slurm/run_grid_goal_dense_exact_eval.slurm
)"

echo "Submitted dense-exact train array: ${train_job}"
echo "Submitted dense-exact eval array:  ${eval_job}"
