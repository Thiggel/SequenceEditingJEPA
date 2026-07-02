#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs

VARIANTS=(
  W_uniform_H0_G_none
  W_uniform_H4_G_none
  W_uniform_H4_16_G_none
  W_uniform_H4_16_32_G_none
  W_inv_sqrt_H0_G_none
  W_inv_sqrt_H4_G_none
  W_inv_sqrt_H4_16_G_none
  W_inv_sqrt_H4_16_32_G_none
  W_gamma_H0_G_none
  W_gamma_H4_G_none
  W_gamma_H4_16_G_none
  W_gamma_H4_16_32_G_none
  G_ic_detached
  G_ic_non_detached
  G_ic_online_no_stopgrad
  G_ic_field_plus_mse
  G_ic_field_only
)

for variant in "${VARIANTS[@]}"; do
  train_job="$(
    sbatch --parsable \
      --job-name="gg_clean17_tr_${variant}" \
      --export=ALL,VARIANT="${variant}" \
      scripts/slurm/run_grid_goal_clean17_train.slurm
  )"
  eval_job="$(
    sbatch --parsable \
      --job-name="gg_clean17_ev_${variant}" \
      --dependency="afterok:${train_job}" \
      --export=ALL,VARIANT="${variant}" \
      scripts/slurm/run_grid_goal_clean17_eval.slurm
  )"
  printf '%s\ttrain=%s\teval=%s\n' "${variant}" "${train_job}" "${eval_job}"
done
