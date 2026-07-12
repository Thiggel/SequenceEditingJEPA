#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs
source scripts/env.sh

DATASETS=(
  wrapped_motion
  rotating_motion
  object_blocked_build
  frontier_build
  random_within_object_build
  interleaved_build
  global_random_build
  completion
  noisy_repair
)
ROWS=(
  "4 0"
  "4 16"
  "8 0"
  "8 2"
  "8 4"
  "8 16"
)
SEEDS=(1707 2707 3707)
RUN_SUFFIX="${RUN_SUFFIX:-rate_transfer_v1_steps${MAX_STEPS:-5000}}"
MANIFEST="${PUZZLE_JEPA_WORK_ROOT}/runs/moving_objects/manifests/${RUN_SUFFIX}.tsv"
mkdir -p "$(dirname "${MANIFEST}")"

if [[ "${SUBMIT:-0}" == "1" ]]; then
  printf 'run_name\tjob_id\tlatent_dim\tmax_objects\tseed\n' > "${MANIFEST}"
fi
previous_stage_ids=""
for data in "${DATASETS[@]}"; do
  stage_ids=()
  for row in "${ROWS[@]}"; do
    read -r latent_dim levels <<<"${row}"
    for seed in "${SEEDS[@]}"; do
      run_name="motion_n8_z${latent_dim}_q${levels}_${RUN_SUFFIX}_${data}_seed${seed}"
      if [[ "${SUBMIT:-0}" == "1" ]]; then
        dependency_args=()
        if [[ -n "${previous_stage_ids}" ]]; then
          dependency_args=(--dependency="afterany:${previous_stage_ids}")
        fi
        job_id="$(
          LATENT_DIM="${latent_dim}" LATENT_QUANTIZATION_LEVELS="${levels}" \
          MIN_OBJECTS=8 MAX_OBJECTS=8 SEED="${seed}" \
          DATA_CONFIG="${data}" OBJECTIVE_CONFIG=ema_vicreg_rate_balanced \
          MAX_STEPS="${MAX_STEPS:-5000}" LEARNING_RATE="${LEARNING_RATE:-3.0e-4}" \
          RUN_SUFFIX="q${levels}_${RUN_SUFFIX}_${data}" \
            sbatch --parsable "${dependency_args[@]}" scripts/slurm/run_moving_objects_train.slurm
        )"
        stage_ids+=("${job_id}")
        printf '%s\t%s\t%s\t8\t%s\n' \
          "${run_name}" "${job_id}" "${latent_dim}" "${seed}" >> "${MANIFEST}"
      else
        printf 'DRY RUN: data=%s z=%s levels=%s N=8 seed=%s run=%s\n' \
          "${data}" "${latent_dim}" "${levels}" "${seed}" "${run_name}"
      fi
    done
  done
  if [[ "${SUBMIT:-0}" == "1" ]]; then
    previous_stage_ids="$(IFS=:; printf '%s' "${stage_ids[*]}")"
  fi
done

if [[ "${SUBMIT:-0}" == "1" ]]; then
  printf 'Manifest: %s (nine dependency-staged families, 18 jobs each)\n' "${MANIFEST}"
else
  printf 'Dry run only. Re-run with SUBMIT=1 to submit 162 selected hard-rate single-CLS jobs.\n'
fi
