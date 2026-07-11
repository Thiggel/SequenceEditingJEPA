#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs
source scripts/env.sh

DATASETS=(reflected_motion wrapped_motion rotating_motion)
SEEDS=(1707 2707 3707)
ROWS=(
  "4 4 ema_vicreg"
  "4 4 ema_vicreg_temporal"
  "4 8 ema_vicreg"
  "4 8 ema_vicreg_temporal"
  "32 4 ema_vicreg"
  "32 8 ema_vicreg"
)
RUN_SUFFIX="${RUN_SUFFIX:-deterministic_confirmation_v1_steps${MAX_STEPS:-5000}}"
MANIFEST="${PUZZLE_JEPA_WORK_ROOT}/runs/moving_objects/manifests/${RUN_SUFFIX}.tsv"
mkdir -p "$(dirname "${MANIFEST}")"

if [[ "${SUBMIT:-0}" == "1" ]]; then
  printf 'run_name\tjob_id\tlatent_dim\tmax_objects\tseed\n' > "${MANIFEST}"
fi
for data in "${DATASETS[@]}"; do
  for row in "${ROWS[@]}"; do
    read -r latent_dim max_objects objective <<<"${row}"
    for seed in "${SEEDS[@]}"; do
      run_name="motion_n${max_objects}_z${latent_dim}_${RUN_SUFFIX}_${data}_${objective}_seed${seed}"
      if [[ "${SUBMIT:-0}" == "1" ]]; then
        job_id="$(
          LATENT_DIM="${latent_dim}" MAX_OBJECTS="${max_objects}" SEED="${seed}" \
          DATA_CONFIG="${data}" OBJECTIVE_CONFIG="${objective}" \
          MAX_STEPS="${MAX_STEPS:-5000}" LEARNING_RATE="${LEARNING_RATE:-3.0e-4}" \
          RUN_SUFFIX="${RUN_SUFFIX}_${data}_${objective}" \
            sbatch --parsable scripts/slurm/run_moving_objects_train.slurm
        )"
        printf '%s\t%s\t%s\t%s\t%s\n' \
          "${run_name}" "${job_id}" "${latent_dim}" "${max_objects}" "${seed}" >> "${MANIFEST}"
      else
        printf 'DRY RUN: data=%s objective=%s z=%s N=%s seed=%s run=%s\n' \
          "${data}" "${objective}" "${latent_dim}" "${max_objects}" "${seed}" "${run_name}"
      fi
    done
  done
done

if [[ "${SUBMIT:-0}" == "1" ]]; then
  printf 'Manifest: %s\n' "${MANIFEST}"
else
  printf 'Dry run only. Re-run with SUBMIT=1 to submit 54 deterministic single-CLS jobs.\n'
fi
