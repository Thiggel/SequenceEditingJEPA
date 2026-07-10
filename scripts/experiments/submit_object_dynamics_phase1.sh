#!/usr/bin/env bash
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p logs

if [[ "${SUBMIT:-0}" == "1" ]]; then
  if [[ "${PRESTAGE_SELECTION_CONFIRMED:-0}" != "1" ]]; then
    printf 'Refusing phase submission: set PRESTAGE_SELECTION_CONFIRMED=1 after reviewing prestage probes.\n' >&2
    exit 2
  fi
  : "${LEARNING_RATE:?Set LEARNING_RATE to the prestage-selected value}"
  : "${MAX_STEPS:?Set MAX_STEPS to the prestage-selected value}"
fi

DATASETS=(
  object_blocked
  frontier_build
  random_within_object
  interleaved_build
  global_random
  noisy_repair
  completion
  transform_identity
  random_off_manifold
)

PHASE1_MODELS=(cls64_r1 cls64_r4 cls64_r8 cls128_r4 cls128_r8 grid128_r8)
PHASE2_OBJECTIVES=(ldad vicreg sigreg ema)
HIERARCHY_MODELS=(h_cls128_h4 h_cls128_h8 h_cls128_h16)

IDS=()
submit_or_print() {
  local data="$1"
  local model="$2"
  local objective="$3"
  local suffix="$4"
  if [[ "${SUBMIT:-0}" == "1" ]]; then
    local job_id
    job_id="$(
      DATA_CONFIG="${data}" \
      MODEL_CONFIG="${model}" \
      OBJECTIVE_CONFIG="${objective}" \
      LEARNING_RATE="${LEARNING_RATE}" \
      MAX_STEPS="${MAX_STEPS}" \
      RUN_SUFFIX="${suffix}" \
        sbatch --parsable scripts/slurm/run_object_dynamics_train.slurm
    )"
    IDS+=("${data}/${model}/${objective}/${suffix}:${job_id}")
  else
    printf 'DRY RUN: DATA_CONFIG=%s MODEL_CONFIG=%s OBJECTIVE_CONFIG=%s LEARNING_RATE=%s MAX_STEPS=%s RUN_SUFFIX=%s sbatch scripts/slurm/run_object_dynamics_train.slurm\n' \
      "${data}" "${model}" "${objective}" "${LEARNING_RATE:-PRESTAGE_REQUIRED}" \
      "${MAX_STEPS:-PRESTAGE_REQUIRED}" "${suffix}"
  fi
}

for data in "${DATASETS[@]}"; do
  for model in "${PHASE1_MODELS[@]}"; do
    submit_or_print "${data}" "${model}" base phase1
  done

  for objective in "${PHASE2_OBJECTIVES[@]}"; do
    submit_or_print "${data}" cls128_r8 "${objective}" phase2
  done
  submit_or_print "${data}" grid128_r8 ldad phase2_grid_ldad

  for model in "${HIERARCHY_MODELS[@]}"; do
    submit_or_print "${data}" "${model}" base phase3
  done
  submit_or_print "${data}" h_cls128_h8 ldad phase3_h8_ldad
  submit_or_print "${data}" h_grid128_h8 ldad phase3_h8_grid_ldad
done

if [[ "${SUBMIT:-0}" == "1" ]]; then
  printf 'Submitted object-dynamics phase sweep jobs:\n'
  printf '  %s\n' "${IDS[@]}"
else
  printf 'Dry run only. Re-run with SUBMIT=1 to submit.\n'
fi
