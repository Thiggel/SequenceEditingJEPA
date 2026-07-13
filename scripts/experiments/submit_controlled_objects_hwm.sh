#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs
source scripts/env.sh

SEEDS=(1707 2707 3707)
ROLLOUTS=(1 2 4 8)
LAMBDAS=(0.75 0.9 0.95 1.0)
DEPTHS=(1 2 3 4)
STRIDES=(2 4 8)

SWEEP_NAME="${SWEEP_NAME:-controlled_hierarchy_rollout_v5_steps${MAX_STEPS:-20000}}"
OUTPUT_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/controlled_objects/${SWEEP_NAME}"
MANIFEST_DIR="${PUZZLE_JEPA_WORK_ROOT}/runs/controlled_objects/manifests"
MANIFEST="${MANIFEST_DIR}/${SWEEP_NAME}.tsv"
mkdir -p "${MANIFEST_DIR}"
printf 'run_name\tjob_id\tblock\tseed\tdepth\tstride\trollout_steps\trollout_all_levels\tlambda\trepresentation\tlatent_dim\tldad_horizon\tldad_weight\tobjective\tdependency\toutput_dir\n' > "${MANIFEST}"

declare -A FLAT_JOBS
declare -A DEPTH_JOBS
declare -A STRIDE_JOBS
declare -A COMBINED_JOBS
JOB_COUNT=0
LAST_JOB_ID=""

submit_one() {
  local run_name="$1"
  local block="$2"
  local seed="$3"
  local depth="$4"
  local stride="$5"
  local rollout="$6"
  local all_levels="$7"
  local lambda="$8"
  local dependency="$9"
  local init_checkpoint="${10}"
  local train_from_level="${11}"
  local job_id
  if [[ "${SUBMIT:-0}" == "1" ]]; then
    local dependency_args=()
    if [[ -n "${dependency}" ]]; then
      dependency_args=("--dependency=afterok:${dependency}")
    fi
    job_id="$(
      sbatch --parsable "${dependency_args[@]}" \
        --export="ALL,RUN_NAME=${run_name},SEED=${seed},OUTPUT_ROOT=${OUTPUT_ROOT},MODEL_CONFIG=cls_hwm,OBJECTIVE_CONFIG=ema_vicreg_strong,HIERARCHY_DEPTH=${depth},HIERARCHY_STRIDE=${stride},ROLLOUT_STEPS=${rollout},ROLLOUT_ALL_LEVELS=${all_levels},ROLLOUT_LAMBDA=${lambda},TRAIN_FROM_LEVEL=${train_from_level},INIT_CHECKPOINT=${init_checkpoint},LATENT_DIM=32,LDAD_HORIZON=1,LDAD_WEIGHT=0,MAX_STEPS=${MAX_STEPS:-20000},BATCH_SIZE=${BATCH_SIZE:-64},PLANNING_EPISODES=${PLANNING_EPISODES:-32},PLANNING_CANDIDATES=${PLANNING_CANDIDATES:-64}" \
        scripts/slurm/run_controlled_objects_train.slurm
    )"
  else
    job_id="dry${JOB_COUNT}"
    printf 'DRY RUN: block=%s run=%s dependency=%s\n' \
      "${block}" "${run_name}" "${dependency:-none}"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tcls\t32\t1\t0\tema_vicreg_strong\t%s\t%s\n' \
    "${run_name}" "${job_id}" "${block}" "${seed}" "${depth}" "${stride}" \
    "${rollout}" "${all_levels}" "${lambda}" "${dependency}" \
    "${OUTPUT_ROOT}/${run_name}" >> "${MANIFEST}"
  JOB_COUNT=$((JOB_COUNT + 1))
  LAST_JOB_ID="${job_id}"
}

# Primitive dense rollout: one independent axis at lambda 1.
for rollout in "${ROLLOUTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run="flat_r${rollout}_l1_seed${seed}"
    submit_one "${run}" rollout "${seed}" 1 4 "${rollout}" false 1.0 "" "" 0
    FLAT_JOBS["${seed},${rollout}"]="${LAST_JOB_ID}"
  done
done

# Lambda is isolated at primitive rollout 4; lambda 1 is reused above.
for lambda in 0.75 0.9 0.95; do
  label="${lambda/./}"
  for seed in "${SEEDS[@]}"; do
    run="flat_r4_l${label}_seed${seed}"
    submit_one "${run}" lambda "${seed}" 1 4 4 false "${lambda}" "" "" 0
  done
done

# Depth at stride 4 is trained one level at a time, freezing every lower level.
for seed in "${SEEDS[@]}"; do
  DEPTH_JOBS["${seed},1"]="${FLAT_JOBS["${seed},1"]}"
  for depth in 2 3 4; do
    previous=$((depth - 1))
    previous_job="${DEPTH_JOBS["${seed},${previous}"]}"
    previous_run="depth${previous}_s4_r1_seed${seed}"
    if [[ "${previous}" == "1" ]]; then
      previous_run="flat_r1_l1_seed${seed}"
    fi
    run="depth${depth}_s4_r1_seed${seed}"
    checkpoint="${OUTPUT_ROOT}/${previous_run}/checkpoint.pt"
    submit_one "${run}" depth "${seed}" "${depth}" 4 1 false 1.0 \
      "${previous_job}" "${checkpoint}" "${previous}"
    DEPTH_JOBS["${seed},${depth}"]="${LAST_JOB_ID}"
  done
done

# Stride at depth 3 uses a private staged depth-2 checkpoint for each stride.
for stride in 2 8; do
  for seed in "${SEEDS[@]}"; do
    base_job="${FLAT_JOBS["${seed},1"]}"
    base_checkpoint="${OUTPUT_ROOT}/flat_r1_l1_seed${seed}/checkpoint.pt"
    stage_run="depth2_s${stride}_r1_seed${seed}"
    submit_one "${stage_run}" stride_stage "${seed}" 2 "${stride}" 1 false 1.0 \
      "${base_job}" "${base_checkpoint}" 1
    STRIDE_JOBS["${seed},${stride},2"]="${LAST_JOB_ID}"

    run="depth3_s${stride}_r1_seed${seed}"
    checkpoint="${OUTPUT_ROOT}/${stage_run}/checkpoint.pt"
    submit_one "${run}" stride "${seed}" 3 "${stride}" 1 false 1.0 \
      "${LAST_JOB_ID}" "${checkpoint}" 2
    STRIDE_JOBS["${seed},${stride},3"]="${LAST_JOB_ID}"
  done
done

# Hierarchy plus rollout compares dense primitive-only and dense every-level loss.
for all_levels in false true; do
  label=low_only
  if [[ "${all_levels}" == "true" ]]; then
    label=all_levels
  fi
  for seed in "${SEEDS[@]}"; do
    base_job="${FLAT_JOBS["${seed},4"]}"
    base_checkpoint="${OUTPUT_ROOT}/flat_r4_l1_seed${seed}/checkpoint.pt"
    stage_run="depth2_s4_r4_${label}_seed${seed}"
    submit_one "${stage_run}" hierarchy_rollout_stage "${seed}" 2 4 4 \
      "${all_levels}" 1.0 "${base_job}" "${base_checkpoint}" 1
    COMBINED_JOBS["${seed},${all_levels},2"]="${LAST_JOB_ID}"

    run="depth3_s4_r4_${label}_seed${seed}"
    checkpoint="${OUTPUT_ROOT}/${stage_run}/checkpoint.pt"
    submit_one "${run}" hierarchy_rollout "${seed}" 3 4 4 "${all_levels}" 1.0 \
      "${LAST_JOB_ID}" "${checkpoint}" 2
    COMBINED_JOBS["${seed},${all_levels},3"]="${LAST_JOB_ID}"
  done
done

if [[ "${JOB_COUNT}" -ne 54 ]]; then
  printf 'Internal error: expected 54 single-CLS jobs, built %s.\n' "${JOB_COUNT}" >&2
  exit 1
fi

if [[ "${SUBMIT:-0}" == "1" ]]; then
  printf 'Submitted %s single-CLS jobs. Manifest: %s\n' "${JOB_COUNT}" "${MANIFEST}"
else
  printf 'Dry run only: %s single-CLS jobs. Re-run with SUBMIT=1 to submit. Manifest: %s\n' \
    "${JOB_COUNT}" "${MANIFEST}"
fi
