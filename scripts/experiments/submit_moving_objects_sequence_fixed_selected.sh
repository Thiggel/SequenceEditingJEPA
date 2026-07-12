#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs
source scripts/env.sh

DATASETS=(
  object_blocked_build
  frontier_build
  random_within_object_build
  interleaved_build
  global_random_build
  completion
  noisy_repair
)
SEEDS=(1707 2707 3707)
ROWS=(
  "2 8 ema_vicreg"
  "4 8 ema_vicreg"
  "4 8 ema_vicreg_temporal"
  "16 8 ema_vicreg"
  "32 4 ema_vicreg"
  "32 8 ema_vicreg"
  "32 8 reconstruction"
  "64 2 ema_vicreg"
)
RUN_SUFFIX="${RUN_SUFFIX:-sequence_fixed_selected_v1_steps${MAX_STEPS:-5000}}"
MANIFEST="${PUZZLE_JEPA_WORK_ROOT}/runs/moving_objects/manifests/${RUN_SUFFIX}.tsv"
mkdir -p "$(dirname "${MANIFEST}")"

if [[ "${SUBMIT:-0}" == "1" ]]; then
  printf 'run_name\tjob_id\tlatent_dim\tmax_objects\tseed\n' > "${MANIFEST}"
fi
previous_stage_ids=""
for data in "${DATASETS[@]}"; do
  stage_ids=()
  for row in "${ROWS[@]}"; do
    read -r latent_dim object_count objective <<<"${row}"
    for seed in "${SEEDS[@]}"; do
      run_name="motion_n${object_count}_z${latent_dim}_${RUN_SUFFIX}_${data}_${objective}_seed${seed}"
      if [[ "${SUBMIT:-0}" == "1" ]]; then
        dependency_args=()
        if [[ -n "${previous_stage_ids}" ]]; then
          dependency_args=(--dependency="afterany:${previous_stage_ids}")
        fi
        job_id="$(
          LATENT_DIM="${latent_dim}" \
          MIN_OBJECTS="${object_count}" MAX_OBJECTS="${object_count}" \
          SEED="${seed}" DATA_CONFIG="${data}" OBJECTIVE_CONFIG="${objective}" \
          MAX_STEPS="${MAX_STEPS:-5000}" LEARNING_RATE="${LEARNING_RATE:-3.0e-4}" \
          RUN_SUFFIX="${RUN_SUFFIX}_${data}_${objective}" \
            sbatch --parsable "${dependency_args[@]}" scripts/slurm/run_moving_objects_train.slurm
        )"
        stage_ids+=("${job_id}")
        printf '%s\t%s\t%s\t%s\t%s\n' \
          "${run_name}" "${job_id}" "${latent_dim}" "${object_count}" "${seed}" \
          >> "${MANIFEST}"
      else
        printf 'DRY RUN: data=%s objective=%s z=%s N=%s seed=%s run=%s\n' \
          "${data}" "${objective}" "${latent_dim}" "${object_count}" "${seed}" "${run_name}"
      fi
    done
  done
  if [[ "${SUBMIT:-0}" == "1" ]]; then
    previous_stage_ids="$(IFS=:; printf '%s' "${stage_ids[*]}")"
  fi
done

if [[ "${SUBMIT:-0}" == "1" ]]; then
  printf 'Manifest: %s (seven dependency-staged families, 24 jobs each)\n' "${MANIFEST}"
else
  printf 'Dry run only. Re-run with SUBMIT=1 to submit 168 exact-load deterministic single-CLS jobs.\n'
fi
