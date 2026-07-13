#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs
source scripts/env.sh

SEEDS=(1707 2707 3707)
TOKEN_DIMS=(128 256)
SWEEP_NAME="${SWEEP_NAME:-controlled_capacity_v6_steps${MAX_STEPS:-20000}}"
OUTPUT_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/controlled_objects/${SWEEP_NAME}"
MANIFEST_DIR="${PUZZLE_JEPA_WORK_ROOT}/runs/controlled_objects/manifests"
MANIFEST="${MANIFEST_DIR}/${SWEEP_NAME}.tsv"
PROBE_MANIFEST="${MANIFEST_DIR}/${SWEEP_NAME}_probes.tsv"
BASELINE_MANIFEST="${BASELINE_MANIFEST:-${MANIFEST_DIR}/controlled_hierarchy_rollout_v5_steps20000.tsv}"
mkdir -p "${MANIFEST_DIR}"
printf 'run_name\tjob_id\tblock\tseed\tdepth\tstride\trollout_steps\trollout_all_levels\tlambda\trepresentation\tlatent_dim\tldad_horizon\tldad_weight\tobjective\tdependency\toutput_dir\ttoken_dim\n' > "${MANIFEST}"
printf 'run_name\tprobe_job_id\ttrain_job_id\tcheckpoint\toutput\n' > "${PROBE_MANIFEST}"

for seed in "${SEEDS[@]}"; do
  for prefix in flat_r4_l1 depth2_s4_r4_low_only; do
    run_name="${prefix}_seed${seed}"
    baseline="$(awk -F '\t' -v run="${run_name}" '$1 == run {print; exit}' "${BASELINE_MANIFEST}")"
    if [[ -z "${baseline}" ]]; then
      printf 'Missing capacity baseline %s in %s.\n' "${run_name}" "${BASELINE_MANIFEST}" >&2
      exit 1
    fi
    block=capacity_flat
    if [[ "${prefix}" == depth2* ]]; then
      block=capacity_hierarchy
    fi
    printf '%s\t64\n' "${baseline}" | awk -F '\t' -v block="${block}" \
      'BEGIN {OFS="\t"} {$3=block; print}' >> "${MANIFEST}"
  done
done

TRAIN_JOB_COUNT=0
PROBE_JOB_COUNT=0
LAST_JOB_ID=""

submit_train() {
  local run_name="$1"
  local block="$2"
  local seed="$3"
  local token_dim="$4"
  local latent_dim="$5"
  local depth="$6"
  local dependency="$7"
  local init_checkpoint="$8"
  local train_from_level="$9"
  local job_id
  if [[ "${SUBMIT:-0}" == "1" ]]; then
    local dependency_args=()
    if [[ -n "${dependency}" ]]; then
      dependency_args=("--dependency=afterok:${dependency}")
    fi
    job_id="$(
      sbatch --parsable "${dependency_args[@]}" \
        --export="ALL,RUN_NAME=${run_name},SEED=${seed},OUTPUT_ROOT=${OUTPUT_ROOT},MODEL_CONFIG=cls_hwm,OBJECTIVE_CONFIG=ema_vicreg_strong,HIERARCHY_DEPTH=${depth},HIERARCHY_STRIDE=4,ROLLOUT_STEPS=4,ROLLOUT_ALL_LEVELS=false,ROLLOUT_LAMBDA=1.0,TRAIN_FROM_LEVEL=${train_from_level},INIT_CHECKPOINT=${init_checkpoint},TOKEN_DIM=${token_dim},LATENT_DIM=${latent_dim},LDAD_HORIZON=1,LDAD_WEIGHT=0,MAX_STEPS=${MAX_STEPS:-20000},BATCH_SIZE=${BATCH_SIZE:-64},PLANNING_EPISODES=${PLANNING_EPISODES:-32},PLANNING_CANDIDATES=${PLANNING_CANDIDATES:-64}" \
        scripts/slurm/run_controlled_objects_train.slurm
    )"
  else
    job_id="dry_train${TRAIN_JOB_COUNT}"
    printf 'DRY RUN: train=%s dependency=%s\n' "${run_name}" "${dependency:-none}"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t4\t4\tfalse\t1.0\tcls\t%s\t1\t0\tema_vicreg_strong\t%s\t%s\t%s\n' \
    "${run_name}" "${job_id}" "${block}" "${seed}" "${depth}" "${latent_dim}" \
    "${dependency}" "${OUTPUT_ROOT}/${run_name}" "${token_dim}" >> "${MANIFEST}"
  TRAIN_JOB_COUNT=$((TRAIN_JOB_COUNT + 1))
  LAST_JOB_ID="${job_id}"
}

submit_probe() {
  local run_name="$1"
  local train_job_id="$2"
  local checkpoint="${OUTPUT_ROOT}/${run_name}/checkpoint.pt"
  local output="${OUTPUT_ROOT}/${run_name}/probe_eval_v2.json"
  local job_id
  if [[ "${SUBMIT:-0}" == "1" ]]; then
    job_id="$(
      sbatch --parsable --dependency="afterok:${train_job_id}" \
        --export="ALL,CHECKPOINT=${checkpoint},OUTPUT_PATH=${output},PROBE_SEED=${PROBE_SEED:-9917},PROBE_TRAIN_SAMPLES=${PROBE_TRAIN_SAMPLES:-1024},PROBE_EVAL_SAMPLES=${PROBE_EVAL_SAMPLES:-512},PROBE_BATCH_SIZE=${PROBE_BATCH_SIZE:-64},PROBE_STEPS=${PROBE_STEPS:-200},PROBE_LEARNING_RATE=${PROBE_LEARNING_RATE:-0.003}" \
        scripts/slurm/run_controlled_objects_probes.slurm
    )"
  else
    job_id="dry_probe${PROBE_JOB_COUNT}"
    printf 'DRY RUN: probe=%s train_job=%s\n' "${run_name}" "${train_job_id}"
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${run_name}" "${job_id}" "${train_job_id}" "${checkpoint}" "${output}" \
    >> "${PROBE_MANIFEST}"
  PROBE_JOB_COUNT=$((PROBE_JOB_COUNT + 1))
}

for token_dim in "${TOKEN_DIMS[@]}"; do
  latent_dim=$((token_dim / 2))
  for seed in "${SEEDS[@]}"; do
    flat="capacity_h${token_dim}_z${latent_dim}_flat_r4_seed${seed}"
    submit_train "${flat}" capacity_flat "${seed}" "${token_dim}" "${latent_dim}" 1 "" "" 0
    flat_job="${LAST_JOB_ID}"
    submit_probe "${flat}" "${flat_job}"

    hierarchy="capacity_h${token_dim}_z${latent_dim}_depth2_s4_r4_seed${seed}"
    checkpoint="${OUTPUT_ROOT}/${flat}/checkpoint.pt"
    submit_train "${hierarchy}" capacity_hierarchy "${seed}" "${token_dim}" \
      "${latent_dim}" 2 "${flat_job}" "${checkpoint}" 1
    submit_probe "${hierarchy}" "${LAST_JOB_ID}"
  done
done

if [[ "${TRAIN_JOB_COUNT}" -ne 12 || "${PROBE_JOB_COUNT}" -ne 12 ]]; then
  printf 'Internal error: expected 12 train and 12 probe jobs, built %s and %s.\n' \
    "${TRAIN_JOB_COUNT}" "${PROBE_JOB_COUNT}" >&2
  exit 1
fi

if [[ "${SUBMIT:-0}" == "1" ]]; then
  printf 'Submitted %s capacity jobs and %s dependent probes. Manifests: %s %s\n' \
    "${TRAIN_JOB_COUNT}" "${PROBE_JOB_COUNT}" "${MANIFEST}" "${PROBE_MANIFEST}"
else
  printf 'Dry run only: %s capacity jobs and %s dependent probes. Re-run with SUBMIT=1. Manifests: %s %s\n' \
    "${TRAIN_JOB_COUNT}" "${PROBE_JOB_COUNT}" "${MANIFEST}" "${PROBE_MANIFEST}"
fi
