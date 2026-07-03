#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs

VARIANTS=(
  K1_uniform
  K1_smooth_count
  K2_uniform
  K2_smooth_count
  K3_uniform
  K3_smooth_count
  K4_uniform
  K4_smooth_count
  K8_uniform
  K8_smooth_count
  K16_uniform
  K16_smooth_count
)

for variant in "${VARIANTS[@]}"; do
  train_job="$(
    sbatch --parsable \
      --job-name="gg_hor_tr_${variant}" \
      --export=ALL,VARIANT="${variant}" \
      scripts/slurm/run_grid_goal_horizon_ablation_train.slurm
  )"
  eval_job="$(
    sbatch --parsable \
      --job-name="gg_hor_ev_${variant}" \
      --dependency="afterok:${train_job}" \
      --export=ALL,VARIANT="${variant}" \
      scripts/slurm/run_grid_goal_horizon_ablation_eval.slurm
  )"
  printf '%s\ttrain=%s\teval=%s\n' "${variant}" "${train_job}" "${eval_job}"
done
