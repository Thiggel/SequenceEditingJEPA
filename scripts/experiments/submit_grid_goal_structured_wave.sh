#!/usr/bin/env bash
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p logs

VARIANTS=(
  S0_cell_baseline
  S1_unit_slots
  S2_global_slot
  S3_progress_slot
  S4_full_slots
  DJ0_action_token_all
  DJ1_marker_all
  DJ2_local_all
  DJ3_cross_all
  DJ4_marker_cell_units
  DJ5_cross_cell_units
  DJ0_action_token_all_single
  DJ1_marker_all_single
  DJ2_local_all_single
  DJ3_cross_all_single
  DJ4_marker_cell_units_single
  DJ5_cross_cell_units_single
  SD0_projection_only
  SD1_progress_rank
  SD2_action_subspace
  SD3_progress_action
  PR0_state_pair
  PR1_legal_listwise
  PR2_counterfactual_successor
  PR3_progress_slot_rank
  PR4_successor_progress
  GW0_terminal_goal
  GW1_waypoint_only
  GW2_waypoint_goal
  GW3_goal_conditioned_waypoint
  GW4_multi_waypoint
)

TRAIN_IDS=()
EVAL_IDS=()

for variant in "${VARIANTS[@]}"; do
  train_id="$(
    VARIANT="${variant}" \
      sbatch --parsable scripts/slurm/run_grid_goal_structured_wave_train.slurm
  )"
  eval_id="$(
    VARIANT="${variant}" \
      sbatch --parsable --dependency="afterok:${train_id}" scripts/slurm/run_grid_goal_structured_wave_eval.slurm
  )"
  TRAIN_IDS+=("${variant}:${train_id}")
  EVAL_IDS+=("${variant}:${eval_id}")
done

printf 'Submitted structured-wave training jobs:\n'
printf '  %s\n' "${TRAIN_IDS[@]}"
printf 'Submitted dependency-held structured-wave eval jobs:\n'
printf '  %s\n' "${EVAL_IDS[@]}"
