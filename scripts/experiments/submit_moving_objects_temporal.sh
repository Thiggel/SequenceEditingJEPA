#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs
source scripts/env.sh

LATENT_DIMS=(4 8 16 32)
MAX_OBJECT_COUNTS=(4 8)
SEEDS=(1707 2707 3707)
RUN_SUFFIX="${RUN_SUFFIX:-temporal_delta_v1_steps${MAX_STEPS:-5000}}"
MANIFEST="${PUZZLE_JEPA_WORK_ROOT}/runs/moving_objects/manifests/${RUN_SUFFIX}.tsv"
mkdir -p "$(dirname "${MANIFEST}")"

if [[ "${SUBMIT:-0}" == "1" ]]; then
  printf 'run_name\tjob_id\tlatent_dim\tmax_objects\tseed\n' > "${MANIFEST}"
fi
for latent_dim in "${LATENT_DIMS[@]}"; do
  for max_objects in "${MAX_OBJECT_COUNTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      run_name="motion_n${max_objects}_z${latent_dim}_${RUN_SUFFIX}_seed${seed}"
      if [[ "${SUBMIT:-0}" == "1" ]]; then
        job_id="$(
          LATENT_DIM="${latent_dim}" MAX_OBJECTS="${max_objects}" SEED="${seed}" \
          MAX_STEPS="${MAX_STEPS:-5000}" LEARNING_RATE="${LEARNING_RATE:-3.0e-4}" \
          OBJECTIVE_CONFIG=ema_vicreg_temporal RUN_SUFFIX="${RUN_SUFFIX}" \
            sbatch --parsable scripts/slurm/run_moving_objects_train.slurm
        )"
        printf '%s\t%s\t%s\t%s\t%s\n' "${run_name}" "${job_id}" "${latent_dim}" "${max_objects}" "${seed}" >> "${MANIFEST}"
      else
        printf 'DRY RUN: temporal_delta latent_dim=%s max_objects=%s seed=%s run=%s\n' \
          "${latent_dim}" "${max_objects}" "${seed}" "${run_name}"
      fi
    done
  done
done

if [[ "${SUBMIT:-0}" == "1" ]]; then
  printf 'Manifest: %s\n' "${MANIFEST}"
else
  printf 'Dry run only. Re-run with SUBMIT=1 to submit 24 single-CLS jobs.\n'
fi
