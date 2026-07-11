#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs
source scripts/env.sh

DATASETS=(wrapped_motion rotating_motion)
OBJECTIVES=(ema_vicreg ema_vicreg_temporal)
SEEDS=(1707 2707 3707)
LATENT_DIM=4
MAX_OBJECTS=8
RUN_SUFFIX="${RUN_SUFFIX:-transfer_v1_steps${MAX_STEPS:-5000}}"
MANIFEST="${PUZZLE_JEPA_WORK_ROOT}/runs/moving_objects/manifests/${RUN_SUFFIX}.tsv"
mkdir -p "$(dirname "${MANIFEST}")"

if [[ "${SUBMIT:-0}" == "1" ]]; then
  printf 'run_name\tjob_id\tlatent_dim\tmax_objects\tseed\n' > "${MANIFEST}"
fi
for data in "${DATASETS[@]}"; do
  for objective in "${OBJECTIVES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      run_name="motion_n8_z4_${RUN_SUFFIX}_${data}_${objective}_seed${seed}"
      if [[ "${SUBMIT:-0}" == "1" ]]; then
        job_id="$(
          LATENT_DIM="${LATENT_DIM}" MAX_OBJECTS="${MAX_OBJECTS}" SEED="${seed}" \
          DATA_CONFIG="${data}" OBJECTIVE_CONFIG="${objective}" \
          MAX_STEPS="${MAX_STEPS:-5000}" LEARNING_RATE="${LEARNING_RATE:-3.0e-4}" \
          RUN_SUFFIX="${RUN_SUFFIX}_${data}_${objective}" \
            sbatch --parsable scripts/slurm/run_moving_objects_train.slurm
        )"
        printf '%s\t%s\t%s\t%s\t%s\n' "${run_name}" "${job_id}" "${LATENT_DIM}" "${MAX_OBJECTS}" "${seed}" >> "${MANIFEST}"
      else
        printf 'DRY RUN: data=%s objective=%s z=4 N=8 seed=%s run=%s\n' \
          "${data}" "${objective}" "${seed}" "${run_name}"
      fi
    done
  done
done

if [[ "${SUBMIT:-0}" == "1" ]]; then
  printf 'Manifest: %s\n' "${MANIFEST}"
else
  printf 'Dry run only. Re-run with SUBMIT=1 to submit 12 single-CLS transfer jobs.\n'
fi
