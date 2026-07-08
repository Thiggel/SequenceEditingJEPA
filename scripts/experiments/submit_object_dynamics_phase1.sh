#!/usr/bin/env bash
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p logs

DATASETS=(
  object_blocked
  frontier_build
  random_within_object
  interleaved_build
  global_random
  noisy_repair
)

PHASE1_MODELS=(cls64_r1 cls64_r4 cls64_r8 cls128_r4 cls128_r8)
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
      RUN_SUFFIX="${suffix}" \
        sbatch --parsable scripts/slurm/run_object_dynamics_train.slurm
    )"
    IDS+=("${data}/${model}/${objective}/${suffix}:${job_id}")
  else
    printf 'DRY RUN: DATA_CONFIG=%s MODEL_CONFIG=%s OBJECTIVE_CONFIG=%s RUN_SUFFIX=%s sbatch scripts/slurm/run_object_dynamics_train.slurm\n' \
      "${data}" "${model}" "${objective}" "${suffix}"
  fi
}

for data in "${DATASETS[@]}"; do
  for model in "${PHASE1_MODELS[@]}"; do
    submit_or_print "${data}" "${model}" base phase1
  done

  for objective in "${PHASE2_OBJECTIVES[@]}"; do
    submit_or_print "${data}" cls128_r8 "${objective}" phase2
  done

  for model in "${HIERARCHY_MODELS[@]}"; do
    submit_or_print "${data}" "${model}" base phase3
  done
  submit_or_print "${data}" h_cls128_h8 ldad phase3_h4_ldad
done

if [[ "${SUBMIT:-0}" == "1" ]]; then
  printf 'Submitted object-dynamics phase sweep jobs:\n'
  printf '  %s\n' "${IDS[@]}"
else
  printf 'Dry run only. Re-run with SUBMIT=1 to submit.\n'
fi
