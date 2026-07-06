#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs

OUTPUT_ROOT="${SINGLE_WIDE_OUTPUT_ROOT:-${WORK:?WORK must be set}/sequence-editing}"

VARIANTS=(
  W0_ema_vicreg_d1024
  W1_ema_ldad_set_d1024
  W2_ldad_vicreg_set_d1024
  W3_ldad_only_set_d1024
)

for variant in "${VARIANTS[@]}"; do
  train_job="$(
    sbatch --parsable \
      --job-name="gg_sw_tr_${variant}" \
      --export=ALL,VARIANT="${variant}",PUZZLE_JEPA_WORK_ROOT="${OUTPUT_ROOT}" \
      scripts/slurm/run_grid_goal_single_wide_train.slurm
  )"
  eval_job="$(
    sbatch --parsable \
      --job-name="gg_sw_ev_${variant}" \
      --dependency="afterok:${train_job}" \
      --export=ALL,VARIANT="${variant}",SKIP_DIAGNOSTICS=1,PUZZLE_JEPA_WORK_ROOT="${OUTPUT_ROOT}" \
      scripts/slurm/run_grid_goal_single_wide_eval.slurm
  )"
  printf 'grid_goal_single_wide\tvariant=%s\ttrain=%s\teval=%s\n' "${variant}" "${train_job}" "${eval_job}"
done
