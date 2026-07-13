#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs
source scripts/env.sh

SEEDS=(1707 2707 3707)
DEPTHS=(1 2 3 4)
STRIDES=(2 4 8)
ROLLOUTS=(1 2 4 8)
LAMBDAS=(0.75 0.9 0.95 1.0)
LDAD_OBJECTIVES=(
  ldad_online
  ldad_ema
  ldad_vicreg_stopgrad
  ldad_vicreg_ema
  ldad_vicreg_online
)
REPRESENTATIONS=(cls grid)

SWEEP_NAME="${SWEEP_NAME:-controlled_hwm_v1_steps${MAX_STEPS:-20000}}"
OUTPUT_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/controlled_objects/${SWEEP_NAME}"
MANIFEST_DIR="${PUZZLE_JEPA_WORK_ROOT}/runs/controlled_objects/manifests"
MANIFEST="${MANIFEST_DIR}/${SWEEP_NAME}.tsv"
mkdir -p "${MANIFEST_DIR}"

declare -A BASE_JOBS
declare -A R4_JOBS
JOB_COUNT=0
LAST_JOB_ID=""

record_header() {
  printf 'run_name\tjob_id\tblock\tseed\tdepth\tstride\trollout_steps\trollout_all_levels\tlambda\trepresentation\tobjective\tdependency\toutput_dir\n' > "${MANIFEST}"
}

submit_one() {
  local run_name="$1"
  local block="$2"
  local seed="$3"
  local depth="$4"
  local stride="$5"
  local rollout="$6"
  local all_levels="$7"
  local lambda="$8"
  local representation="$9"
  local objective="${10}"
  local dependency="${11}"
  local init_checkpoint="${12}"
  local train_from_level="${13}"
  local model_config=cls_hwm
  local job_id
  if [[ "${representation}" == "grid" ]]; then
    model_config=grid_ldad
  fi
  if [[ "${SUBMIT:-0}" == "1" ]]; then
    local dependency_args=()
    if [[ -n "${dependency}" ]]; then
      dependency_args=("--dependency=afterok:${dependency}")
    fi
    job_id="$(
      sbatch --parsable "${dependency_args[@]}" \
        --export="ALL,RUN_NAME=${run_name},SEED=${seed},OUTPUT_ROOT=${OUTPUT_ROOT},MODEL_CONFIG=${model_config},OBJECTIVE_CONFIG=${objective},HIERARCHY_DEPTH=${depth},HIERARCHY_STRIDE=${stride},ROLLOUT_STEPS=${rollout},ROLLOUT_ALL_LEVELS=${all_levels},ROLLOUT_LAMBDA=${lambda},TRAIN_FROM_LEVEL=${train_from_level},INIT_CHECKPOINT=${init_checkpoint},MAX_STEPS=${MAX_STEPS:-20000},BATCH_SIZE=${BATCH_SIZE:-64}" \
        scripts/slurm/run_controlled_objects_train.slurm
    )"
  else
    job_id="dry${JOB_COUNT}"
    printf 'DRY RUN: block=%s run=%s dependency=%s\n' "${block}" "${run_name}" "${dependency:-none}"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${run_name}" "${job_id}" "${block}" "${seed}" "${depth}" "${stride}" \
    "${rollout}" "${all_levels}" "${lambda}" "${representation}" "${objective}" \
    "${dependency}" "${OUTPUT_ROOT}/${run_name}" >> "${MANIFEST}"
  JOB_COUNT=$((JOB_COUNT + 1))
  LAST_JOB_ID="${job_id}"
}

record_header

# Shared flat H1 checkpoints: hierarchy depth-1 and rollout-1 baselines.
for seed in "${SEEDS[@]}"; do
  run="depth1_s4_r1_l1_seed${seed}"
  submit_one "${run}" depth "${seed}" 1 4 1 false 1.0 cls ema_vicreg "" "" 0
  BASE_JOBS["${seed}"]="${LAST_JOB_ID}"
done

# (1) Hierarchy depth at fixed stride 4. Depth 1 is the shared baseline above.
for depth in 2 3 4; do
  for seed in "${SEEDS[@]}"; do
    run="depth${depth}_s4_r1_l1_seed${seed}"
    checkpoint="${OUTPUT_ROOT}/depth1_s4_r1_l1_seed${seed}/checkpoint.pt"
    submit_one "${run}" depth "${seed}" "${depth}" 4 1 false 1.0 cls ema_vicreg \
      "${BASE_JOBS[$seed]}" "${checkpoint}" 1
  done
done

# (2) Hierarchy stride at fixed depth 3. Stride 4 is reused from the depth block.
for stride in 2 8; do
  for seed in "${SEEDS[@]}"; do
    run="depth3_s${stride}_r1_l1_seed${seed}"
    checkpoint="${OUTPUT_ROOT}/depth1_s4_r1_l1_seed${seed}/checkpoint.pt"
    submit_one "${run}" stride "${seed}" 3 "${stride}" 1 false 1.0 cls ema_vicreg \
      "${BASE_JOBS[$seed]}" "${checkpoint}" 1
  done
done

# (3) Primitive multi-step rollout. Rollout 1 is the shared baseline.
for rollout in 2 4 8; do
  for seed in "${SEEDS[@]}"; do
    run="flat_r${rollout}_l1_seed${seed}"
    submit_one "${run}" rollout "${seed}" 1 4 "${rollout}" false 1.0 cls ema_vicreg "" "" 0
    if [[ "${rollout}" == "4" ]]; then
      R4_JOBS["${seed}"]="${LAST_JOB_ID}"
    fi
  done
done

# (4) Hierarchy + four-step rollout: dense primitive only versus every level.
for all_levels in false true; do
  label=low_only
  if [[ "${all_levels}" == "true" ]]; then
    label=all_levels
  fi
  for seed in "${SEEDS[@]}"; do
    run="depth3_s4_r4_${label}_l09_seed${seed}"
    checkpoint="${OUTPUT_ROOT}/flat_r4_l1_seed${seed}/checkpoint.pt"
    submit_one "${run}" hierarchy_rollout "${seed}" 3 4 4 "${all_levels}" 0.9 cls \
      ema_vicreg "${R4_JOBS[$seed]}" "${checkpoint}" 1
  done
done

# (5) Lambda is isolated at flat rollout 4. Lambda 1 reuses flat_r4_l1.
for lambda in 0.75 0.9 0.95; do
  label="${lambda/./}"
  for seed in "${SEEDS[@]}"; do
    run="flat_r4_l${label}_seed${seed}"
    submit_one "${run}" lambda "${seed}" 1 4 4 false "${lambda}" cls ema_vicreg \
      "" "" 0
  done
done

# (6) Delta-JEPA factorization. Every objective has paired CLS and full-grid rows.
for objective in "${LDAD_OBJECTIVES[@]}"; do
  for representation in "${REPRESENTATIONS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      run="${objective}_${representation}_r4_seed${seed}"
      submit_one "${run}" ldad "${seed}" 1 4 4 false 1.0 "${representation}" \
        "${objective}" "" "" 0
    done
  done
done

if [[ "${JOB_COUNT}" -ne 72 ]]; then
  printf 'Internal error: expected 72 unique jobs, built %s.\n' "${JOB_COUNT}" >&2
  exit 1
fi

if [[ "${SUBMIT:-0}" == "1" ]]; then
  printf 'Submitted %s jobs. Manifest: %s\n' "${JOB_COUNT}" "${MANIFEST}"
else
  printf 'Dry run only: %s jobs. Re-run with SUBMIT=1 to submit. Manifest: %s\n' \
    "${JOB_COUNT}" "${MANIFEST}"
fi
