#!/usr/bin/env bash
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p logs
source scripts/env.sh

if [[ "${SUBMIT:-0}" == "1" ]]; then
  if [[ "${PRESTAGE_SELECTION_CONFIRMED:-0}" != "1" ]]; then
    printf 'Refusing phase submission: set PRESTAGE_SELECTION_CONFIRMED=1 after reviewing prestage probes.\n' >&2
    exit 2
  fi
  : "${LEARNING_RATE:?Set LEARNING_RATE to the prestage-selected value}"
  : "${MAX_STEPS:?Set MAX_STEPS to the prestage-selected value}"
fi

SEEDS=(1707 2707 3707)
if [[ -n "${SEEDS_OVERRIDE:-}" ]]; then
  read -r -a SEEDS <<< "${SEEDS_OVERRIDE}"
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
  local seed="$5"
  LAST_JOB_ID=""
  if [[ "${SUBMIT:-0}" == "1" ]]; then
    local job_id
    job_id="$(
      DATA_CONFIG="${data}" \
      MODEL_CONFIG="${model}" \
      OBJECTIVE_CONFIG="${objective}" \
      SEED="${seed}" \
      LEARNING_RATE="${LEARNING_RATE}" \
      MAX_STEPS="${MAX_STEPS}" \
      EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-1000}" \
      RUN_SUFFIX="${suffix}_seed${seed}" \
        sbatch --parsable scripts/slurm/run_object_dynamics_train.slurm
    )"
    IDS+=("${data}/${model}/${objective}/${suffix}/seed${seed}:${job_id}")
    LAST_JOB_ID="${job_id}"
  else
    printf 'DRY RUN: DATA_CONFIG=%s MODEL_CONFIG=%s OBJECTIVE_CONFIG=%s SEED=%s LEARNING_RATE=%s MAX_STEPS=%s EVAL_EVERY_STEPS=%s RUN_SUFFIX=%s sbatch scripts/slurm/run_object_dynamics_train.slurm\n' \
      "${data}" "${model}" "${objective}" "${seed}" "${LEARNING_RATE:-PRESTAGE_REQUIRED}" \
      "${MAX_STEPS:-PRESTAGE_REQUIRED}" "${EVAL_EVERY_STEPS:-1000}" "${suffix}_seed${seed}"
  fi
}

submit_staged_hwm() {
  local data="$1"
  local seed="$2"
  local low_job="$3"
  local low_checkpoint="$4"
  local suffix="phase3_h8_staged_seed${seed}"
  if [[ "${SUBMIT:-0}" == "1" ]]; then
    local job_id
    job_id="$(
      DATA_CONFIG="${data}" \
      MODEL_CONFIG=h_cls128_h8 \
      OBJECTIVE_CONFIG=base \
      SEED="${seed}" \
      LEARNING_RATE="${LEARNING_RATE}" \
      MAX_STEPS="${MAX_STEPS}" \
      EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-1000}" \
      INITIAL_CHECKPOINT="${low_checkpoint}" \
      TRAINABLE_COMPONENTS=hierarchy_only \
      RUN_SUFFIX="${suffix}" \
        sbatch --parsable --dependency="afterok:${low_job}" scripts/slurm/run_object_dynamics_train.slurm
    )"
    IDS+=("${data}/h_cls128_h8/base/staged/seed${seed}:${job_id}")
  else
    printf 'DRY RUN: DATA_CONFIG=%s MODEL_CONFIG=h_cls128_h8 OBJECTIVE_CONFIG=base SEED=%s LEARNING_RATE=%s MAX_STEPS=%s EVAL_EVERY_STEPS=%s INITIAL_CHECKPOINT=%s TRAINABLE_COMPONENTS=hierarchy_only RUN_SUFFIX=%s sbatch --dependency=afterok:<cls128_r8_base_job> scripts/slurm/run_object_dynamics_train.slurm\n' \
      "${data}" "${seed}" "${LEARNING_RATE:-PRESTAGE_REQUIRED}" "${MAX_STEPS:-PRESTAGE_REQUIRED}" \
      "${EVAL_EVERY_STEPS:-1000}" "${low_checkpoint}" "${suffix}"
  fi
}

for seed in "${SEEDS[@]}"; do
  for data in "${DATASETS[@]}"; do
    low_job=""
    for model in "${PHASE1_MODELS[@]}"; do
      submit_or_print "${data}" "${model}" base phase1 "${seed}"
      if [[ "${model}" == "cls128_r8" ]]; then
        low_job="${LAST_JOB_ID}"
      fi
    done

    for objective in "${PHASE2_OBJECTIVES[@]}"; do
      submit_or_print "${data}" cls128_r8 "${objective}" phase2 "${seed}"
    done
    submit_or_print "${data}" grid128_r8 ldad phase2_grid_ldad "${seed}"

    for model in "${HIERARCHY_MODELS[@]}"; do
      submit_or_print "${data}" "${model}" base phase3 "${seed}"
    done
    submit_or_print "${data}" h_cls128_h8 ldad phase3_h8_ldad "${seed}"
    submit_or_print "${data}" h_grid128_h8 ldad phase3_h8_grid_ldad "${seed}"
    output_root="${PUZZLE_JEPA_WORK_ROOT:-${WORK:-/tmp}/sequence-editing-object-dynamics-20260708}"
    low_checkpoint="${output_root}/runs/object_dynamics/${data}_cls128_r8_base_phase1_seed${seed}/checkpoint.pt"
    submit_staged_hwm "${data}" "${seed}" "${low_job}" "${low_checkpoint}"
    submit_or_print "${data}" cls128_r8 reconstruction control_reconstruction "${seed}"
  done
done

if [[ "${SUBMIT:-0}" == "1" ]]; then
  printf 'Submitted object-dynamics phase sweep jobs:\n'
  printf '  %s\n' "${IDS[@]}"
else
  printf 'Dry run only. Re-run with SUBMIT=1 to submit.\n'
fi
