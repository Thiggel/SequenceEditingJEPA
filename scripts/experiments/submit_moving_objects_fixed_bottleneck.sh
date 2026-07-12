#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs
source scripts/env.sh

LATENT_DIMS=(2 4 8 16 32 64)
OBJECT_COUNTS=(1 2 4 6 8)
SEEDS=(1707 2707 3707)
RUN_SUFFIX="${RUN_SUFFIX:-fixed_load_reflected_v1_steps${MAX_STEPS:-5000}}"
MANIFEST="${PUZZLE_JEPA_WORK_ROOT}/runs/moving_objects/manifests/${RUN_SUFFIX}.tsv"
mkdir -p "$(dirname "${MANIFEST}")"

if [[ "${SUBMIT:-0}" == "1" ]]; then
  printf 'run_name\tjob_id\tlatent_dim\tmax_objects\tseed\n' > "${MANIFEST}"
fi
for latent_dim in "${LATENT_DIMS[@]}"; do
  for object_count in "${OBJECT_COUNTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      run_name="motion_n${object_count}_z${latent_dim}_${RUN_SUFFIX}_seed${seed}"
      if [[ "${SUBMIT:-0}" == "1" ]]; then
        job_id="$(
          LATENT_DIM="${latent_dim}" \
          MIN_OBJECTS="${object_count}" MAX_OBJECTS="${object_count}" \
          SEED="${seed}" MAX_STEPS="${MAX_STEPS:-5000}" \
          LEARNING_RATE="${LEARNING_RATE:-3.0e-4}" RUN_SUFFIX="${RUN_SUFFIX}" \
            sbatch --parsable scripts/slurm/run_moving_objects_train.slurm
        )"
        printf '%s\t%s\t%s\t%s\t%s\n' \
          "${run_name}" "${job_id}" "${latent_dim}" "${object_count}" "${seed}" \
          >> "${MANIFEST}"
      else
        printf 'DRY RUN: latent_dim=%s objects=%s seed=%s run=%s\n' \
          "${latent_dim}" "${object_count}" "${seed}" "${run_name}"
      fi
    done
  done
done

if [[ "${SUBMIT:-0}" == "1" ]]; then
  printf 'Manifest: %s\n' "${MANIFEST}"
else
  printf 'Dry run only. Re-run with SUBMIT=1 to submit 90 exact-load deterministic single-CLS jobs.\n'
fi
