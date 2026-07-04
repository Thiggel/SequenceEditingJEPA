#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs

RUN_SUFFIX="${RUN_SUFFIX:-_mb4ga2}"
VARIANTS=(
  E5_waypoint_h16_hierarchy
  V3_waypoint_asym_hindsight
  I0_integrated_waypoint_asym
  I1_integrated_waypoint_iql
  I2_integrated_best_delta_if_gate_passes_grid
  I2_integrated_best_delta_if_gate_passes_single
)

EVAL_MODES=(
  waypoint_cem
  waypoint_mppi
)

for variant in "${VARIANTS[@]}"; do
  for mode in "${EVAL_MODES[@]}"; do
    job="$(
      sbatch --parsable \
        --job-name="gg_wpm_${variant}_${mode}" \
        --export=ALL,VARIANT="${variant}",RUN_SUFFIX="${RUN_SUFFIX}",GRID_GOAL_WEEKEND_RUN_SUFFIX="${RUN_SUFFIX}",EVAL_MODE="${mode}",SKIP_DIAGNOSTICS=1 \
        scripts/slurm/run_grid_goal_weekend_eval.slurm
    )"
    printf 'grid_goal_weekend_waypoint_macro\tvariant=%s\tmode=%s\tjob=%s\n' "${variant}" "${mode}" "${job}"
  done
done
