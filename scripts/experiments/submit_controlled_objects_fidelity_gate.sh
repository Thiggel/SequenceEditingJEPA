#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs
source scripts/env.sh

SEEDS=(1707 2707 3707)
LATENT_DIMS=(4 8 16 32)
BASE_OBJECTIVES=(ema_vicreg ema_vicreg_strong)
LDAD_OBJECTIVES=(
  ldad_online
  ldad_ema
  ldad_vicreg_stopgrad
  ldad_vicreg_ema
  ldad_vicreg_online
)
REPRESENTATIONS=(cls grid)

SWEEP_NAME="${SWEEP_NAME:-controlled_fidelity_v2_steps${MAX_STEPS:-5000}}"
OUTPUT_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/controlled_objects/${SWEEP_NAME}"
MANIFEST_DIR="${PUZZLE_JEPA_WORK_ROOT}/runs/controlled_objects/manifests"
MANIFEST="${MANIFEST_DIR}/${SWEEP_NAME}.tsv"
mkdir -p "${MANIFEST_DIR}"
printf 'run_name\tjob_id\tblock\tseed\tdepth\tstride\trollout_steps\trollout_all_levels\tlambda\trepresentation\tlatent_dim\tobjective\tdependency\toutput_dir\n' > "${MANIFEST}"

JOB_COUNT=0

submit_one() {
  local run_name="$1"
  local block="$2"
  local seed="$3"
  local representation="$4"
  local objective="$5"
  local latent_dim="$6"
  local model_config=cls_hwm
  local job_id
  if [[ "${representation}" == "grid" ]]; then
    model_config=grid_ldad
  fi
  if [[ "${SUBMIT:-0}" == "1" ]]; then
    job_id="$(
      sbatch --parsable \
        --export="ALL,RUN_NAME=${run_name},SEED=${seed},OUTPUT_ROOT=${OUTPUT_ROOT},MODEL_CONFIG=${model_config},OBJECTIVE_CONFIG=${objective},HIERARCHY_DEPTH=1,HIERARCHY_STRIDE=4,ROLLOUT_STEPS=4,ROLLOUT_ALL_LEVELS=false,ROLLOUT_LAMBDA=1.0,TRAIN_FROM_LEVEL=0,LATENT_DIM=${latent_dim},MAX_STEPS=${MAX_STEPS:-5000},BATCH_SIZE=${BATCH_SIZE:-64},PLANNING_EPISODES=${PLANNING_EPISODES:-8},PLANNING_CANDIDATES=${PLANNING_CANDIDATES:-32}" \
        scripts/slurm/run_controlled_objects_train.slurm
    )"
  else
    job_id="dry${JOB_COUNT}"
    printf 'DRY RUN: block=%s run=%s latent_dim=%s\n' \
      "${block}" "${run_name}" "${latent_dim}"
  fi
  printf '%s\t%s\t%s\t%s\t1\t4\t4\tfalse\t1.0\t%s\t%s\t%s\t\t%s\n' \
    "${run_name}" "${job_id}" "${block}" "${seed}" "${representation}" \
    "${latent_dim}" "${objective}" "${OUTPUT_ROOT}/${run_name}" >> "${MANIFEST}"
  JOB_COUNT=$((JOB_COUNT + 1))
}

# State-bottleneck gate under the conventional EMA target objective.
for objective in "${BASE_OBJECTIVES[@]}"; do
  for latent_dim in "${LATENT_DIMS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      run="bottleneck_${objective}_z${latent_dim}_seed${seed}"
      submit_one "${run}" bottleneck "${seed}" cls "${objective}" "${latent_dim}"
    done
  done
done

# Corrected Delta-JEPA factorization. Every objective remains CLS/grid paired.
for objective in "${LDAD_OBJECTIVES[@]}"; do
  for representation in "${REPRESENTATIONS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      run="${objective}_${representation}_z32_seed${seed}"
      submit_one "${run}" ldad "${seed}" "${representation}" "${objective}" 32
    done
  done
done

if [[ "${JOB_COUNT}" -ne 54 ]]; then
  printf 'Internal error: expected 54 jobs, built %s.\n' "${JOB_COUNT}" >&2
  exit 1
fi

if [[ "${SUBMIT:-0}" == "1" ]]; then
  printf 'Submitted %s jobs. Manifest: %s\n' "${JOB_COUNT}" "${MANIFEST}"
else
  printf 'Dry run only: %s jobs. Re-run with SUBMIT=1 to submit. Manifest: %s\n' \
    "${JOB_COUNT}" "${MANIFEST}"
fi
