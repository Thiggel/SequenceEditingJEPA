#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs

VARIANTS=(
  A_anchor_repro
  A_refactor_equiv_14816
  A_refactor_equiv_14816_dropout_off
  A_smooth_14816_like
  A_uniform_k16
  A_inv_sqrt_k16
  A_gamma_k16
  A_inv_sqrt_k8
  A_old_path_h16_only
  A_old_path_h8_only
  A_no_goal_mse
  A_initial_current_goal
  A_no_hierarchy
  A_no_predict_delta
)

for variant in "${VARIANTS[@]}"; do
  train_job="$(
    sbatch --parsable \
      --job-name="gg_mfac_tr_${variant}" \
      --export=ALL,VARIANT="${variant}" \
      scripts/slurm/run_grid_goal_minaux_factor_train.slurm
  )"
  eval_job="$(
    sbatch --parsable \
      --job-name="gg_mfac_ev_${variant}" \
      --dependency="afterok:${train_job}" \
      --export=ALL,VARIANT="${variant}" \
      scripts/slurm/run_grid_goal_minaux_factor_eval.slurm
  )"
  printf '%s\ttrain=%s\teval=%s\n' "${variant}" "${train_job}" "${eval_job}"
done
