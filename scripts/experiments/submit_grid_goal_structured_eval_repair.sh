#!/usr/bin/env bash
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p logs

VARIANTS=(
  S1_unit_slots
  S2_global_slot
  S3_progress_slot
  S4_full_slots
  DJ4_marker_cell_units
  DJ5_cross_cell_units
  SD0_projection_only
  SD1_progress_rank
  SD2_action_subspace
  SD3_progress_action
  PR0_state_pair
  GW1_waypoint_only
  C0_full_ldad_sd
  C2_full_sd_pr
)

IDS=()
for variant in "${VARIANTS[@]}"; do
  if [[ "${SUBMIT:-0}" == "1" ]]; then
    job_id="$(
      VARIANT="${variant}" \
      SKIP_PROBES=1 \
      EVAL_OUTPUT_SUFFIX="_mask_repair_20260710" \
        sbatch --parsable --time=12:00:00 scripts/slurm/run_grid_goal_structured_wave_eval.slurm
    )"
    IDS+=("${variant}:${job_id}")
  else
    printf 'DRY RUN: VARIANT=%s SKIP_PROBES=1 EVAL_OUTPUT_SUFFIX=_mask_repair_20260710 sbatch --time=12:00:00 scripts/slurm/run_grid_goal_structured_wave_eval.slurm\n' \
      "${variant}"
  fi
done

if [[ "${SUBMIT:-0}" == "1" ]]; then
  printf 'Submitted structured-mask repair eval jobs:\n'
  printf '  %s\n' "${IDS[@]}"
else
  printf 'Dry run only. Re-run with SUBMIT=1 to submit.\n'
fi
