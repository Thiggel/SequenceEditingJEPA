#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs

VARIANTS=(
  D4_H4_16
  D8_H4_16
  D16_H4_16
  D8_H4_16_32
)

EVAL_MODES=(
  baseline
  cem_none
  cem_codebook
  mppi_none
  mppi_codebook
)

for variant in "${VARIANTS[@]}"; do
  train_job="$(
    sbatch --parsable \
      --job-name="gg_hwm_tr_${variant}" \
      --export=ALL,VARIANT="${variant}" \
      scripts/slurm/run_grid_goal_macro_hwm_train.slurm
  )"
  printf '%s\ttrain=%s\n' "${variant}" "${train_job}"
  for eval_mode in "${EVAL_MODES[@]}"; do
    eval_job="$(
      sbatch --parsable \
        --job-name="gg_hwm_ev_${variant}_${eval_mode}" \
        --dependency="afterok:${train_job}" \
        --export=ALL,VARIANT="${variant}",EVAL_MODE="${eval_mode}" \
        scripts/slurm/run_grid_goal_macro_hwm_eval.slurm
    )"
    printf '%s\t%s_eval=%s\n' "${variant}" "${eval_mode}" "${eval_job}"
  done
done
