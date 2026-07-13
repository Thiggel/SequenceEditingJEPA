#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs
source scripts/env.sh

SEEDS=(1707 2707 3707)
REPRESENTATIONS=(cls grid)

SWEEP_NAME="${SWEEP_NAME:-controlled_delta_long_v4_steps${MAX_STEPS:-20000}}"
OUTPUT_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/controlled_objects/${SWEEP_NAME}"
MANIFEST_DIR="${PUZZLE_JEPA_WORK_ROOT}/runs/controlled_objects/manifests"
MANIFEST="${MANIFEST_DIR}/${SWEEP_NAME}.tsv"
mkdir -p "${MANIFEST_DIR}"
printf 'run_name\tjob_id\tblock\tseed\tdepth\tstride\trollout_steps\trollout_all_levels\tlambda\trepresentation\tlatent_dim\tldad_horizon\tldad_weight\tobjective\tdependency\toutput_dir\n' > "${MANIFEST}"

JOB_COUNT=0
for representation in "${REPRESENTATIONS[@]}"; do
  model_config=cls_hwm
  if [[ "${representation}" == "grid" ]]; then
    model_config=grid_ldad
  fi
  for seed in "${SEEDS[@]}"; do
    run="delta_h4_w1_${representation}_seed${seed}"
    if [[ "${SUBMIT:-0}" == "1" ]]; then
      job_id="$(
        sbatch --parsable \
          --export="ALL,RUN_NAME=${run},SEED=${seed},OUTPUT_ROOT=${OUTPUT_ROOT},MODEL_CONFIG=${model_config},OBJECTIVE_CONFIG=ldad_online,HIERARCHY_DEPTH=1,HIERARCHY_STRIDE=4,ROLLOUT_STEPS=4,ROLLOUT_ALL_LEVELS=false,ROLLOUT_LAMBDA=1.0,TRAIN_FROM_LEVEL=0,LATENT_DIM=32,LDAD_HORIZON=4,LDAD_WEIGHT=1,MAX_STEPS=${MAX_STEPS:-20000},BATCH_SIZE=${BATCH_SIZE:-64},PLANNING_EPISODES=${PLANNING_EPISODES:-32},PLANNING_CANDIDATES=${PLANNING_CANDIDATES:-64}" \
          scripts/slurm/run_controlled_objects_train.slurm
      )"
    else
      job_id="dry${JOB_COUNT}"
      printf 'DRY RUN: h=4 weight=1 latent=%s run=%s\n' \
        "${representation}" "${run}"
    fi
    printf '%s\t%s\tdelta_long\t%s\t1\t4\t4\tfalse\t1.0\t%s\t32\t4\t1\tldad_online\t\t%s\n' \
      "${run}" "${job_id}" "${seed}" "${representation}" \
      "${OUTPUT_ROOT}/${run}" >> "${MANIFEST}"
    JOB_COUNT=$((JOB_COUNT + 1))
  done
done

if [[ "${JOB_COUNT}" -ne 6 ]]; then
  printf 'Internal error: expected 6 jobs, built %s.\n' "${JOB_COUNT}" >&2
  exit 1
fi

if [[ "${SUBMIT:-0}" == "1" ]]; then
  printf 'Submitted %s jobs. Manifest: %s\n' "${JOB_COUNT}" "${MANIFEST}"
else
  printf 'Dry run only: %s jobs. Re-run with SUBMIT=1 to submit. Manifest: %s\n' \
    "${JOB_COUNT}" "${MANIFEST}"
fi
