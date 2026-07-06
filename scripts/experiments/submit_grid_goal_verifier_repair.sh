#!/usr/bin/env bash
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p logs

OUTPUT_ROOT="${VERIFIER_REPAIR_OUTPUT_ROOT:-${WORK:?WORK must be set}/sequence-editing-repair-20260706}"
RUN_SUFFIX="${RUN_SUFFIX:-_repair}"

VARIANTS=(
  B0_current_E4
  B1_current_F0
  S1_balanced_partials
  S2_near_solution
  S3_recovery_states
  S4_hard_action_sets
  W1_energy_x3
  W2_energy_x10
  W3_energy_x30
  W4_rank_x3
  W5_rank_x10
  G1_energy_projection
  G2_energy_finetune
  G3_dyn_half
  P1_pred_h1
  P2_pred_h4
  P3_pred_h8
  P4_score_consistency
  L1_pair_same_cell
  L2_pair_overwrite
  L3_listwise_32
  L4_listwise_128
  L5_exhaustive_small
  C1_sampling_weight
  C2_sampling_rank
  C3_pred_rank
  C4_energy_proj_rank
  C5_best_no_policy
  C6_best_policy
  C7_finetune_best
  C8_full
)

TRAIN_JOBS=()
EVAL_JOBS=()
for variant in "${VARIANTS[@]}"; do
  train_id="$(
    sbatch --parsable \
      --job-name="gg_verrep_tr_${variant}" \
      --export=ALL,VARIANT="${variant}",RUN_SUFFIX="${RUN_SUFFIX}",PUZZLE_JEPA_WORK_ROOT="${OUTPUT_ROOT}" \
      scripts/slurm/run_grid_goal_verifier_repair_train.slurm
  )"
  TRAIN_JOBS+=("${variant}:${train_id}")
  eval_id="$(
    sbatch --parsable \
      --job-name="gg_verrep_ev_${variant}" \
      --dependency="afterok:${train_id}" \
      --export=ALL,VARIANT="${variant}",RUN_SUFFIX="${RUN_SUFFIX}",PUZZLE_JEPA_WORK_ROOT="${OUTPUT_ROOT}" \
      scripts/slurm/run_grid_goal_verifier_repair_eval.slurm
  )"
  EVAL_JOBS+=("${variant}:${eval_id}")
done

printf 'Submitted verifier-energy repair training jobs to %s:\n' "${OUTPUT_ROOT}"
printf '  %s\n' "${TRAIN_JOBS[@]}"
printf 'Submitted dependency-held verifier-energy repair eval jobs:\n'
printf '  %s\n' "${EVAL_JOBS[@]}"
