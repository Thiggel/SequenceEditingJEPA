#!/usr/bin/env bash
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p logs

VARIANTS=(
  E0_base_oracle_sanity
  E1_compat_state
  E2_remaining_state
  E3_wr_state
  E4_wr_predicted
  E5_wr_pairwise_rank
  E6_wr_listwise_rank
  E7_wr_listwise_policy
  E8_wr_no_counterfactual
  E9_wr_no_corruption
  F0_full_score
  F1_full_policy
)

TRAIN_JOBS=()
EVAL_JOBS=()
for variant in "${VARIANTS[@]}"; do
  train_id="$(
    VARIANT="${variant}" \
    RUN_SUFFIX="${RUN_SUFFIX:-}" \
    sbatch --parsable scripts/slurm/run_grid_goal_verifier_energy_train.slurm
  )"
  TRAIN_JOBS+=("${variant}:${train_id}")
  eval_id="$(
    VARIANT="${variant}" \
    RUN_SUFFIX="${RUN_SUFFIX:-}" \
    sbatch --parsable --dependency="afterok:${train_id}" scripts/slurm/run_grid_goal_verifier_energy_eval.slurm
  )"
  EVAL_JOBS+=("${variant}:${eval_id}")
done

printf 'Submitted verifier-energy training jobs:\n'
printf '  %s\n' "${TRAIN_JOBS[@]}"
printf 'Submitted dependency-held verifier-energy eval jobs:\n'
printf '  %s\n' "${EVAL_JOBS[@]}"
