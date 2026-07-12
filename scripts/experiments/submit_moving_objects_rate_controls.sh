#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs
source scripts/env.sh

ROWS=(
  "4 16 occupancy_rate_balanced"
  "4 16 reconstruction_rate_balanced"
  "8 16 occupancy_rate_balanced"
  "8 16 reconstruction_rate_balanced"
)
SEEDS=(1707 2707 3707)
RUN_SUFFIX="${RUN_SUFFIX:-rate_controls_v1_steps${MAX_STEPS:-5000}}"
MANIFEST="${PUZZLE_JEPA_WORK_ROOT}/runs/moving_objects/manifests/${RUN_SUFFIX}.tsv"
mkdir -p "$(dirname "${MANIFEST}")"

if [[ "${SUBMIT:-0}" == "1" ]]; then
  printf 'run_name\tjob_id\tlatent_dim\tmax_objects\tseed\n' > "${MANIFEST}"
fi
for row in "${ROWS[@]}"; do
  read -r latent_dim levels objective <<<"${row}"
  for seed in "${SEEDS[@]}"; do
    run_name="motion_n8_z${latent_dim}_q${levels}_${RUN_SUFFIX}_${objective}_seed${seed}"
    if [[ "${SUBMIT:-0}" == "1" ]]; then
      job_id="$(
        LATENT_DIM="${latent_dim}" LATENT_QUANTIZATION_LEVELS="${levels}" \
        MIN_OBJECTS=8 MAX_OBJECTS=8 SEED="${seed}" \
        DATA_CONFIG=reflected_motion OBJECTIVE_CONFIG="${objective}" \
        MAX_STEPS="${MAX_STEPS:-5000}" LEARNING_RATE="${LEARNING_RATE:-3.0e-4}" \
        RUN_SUFFIX="q${levels}_${RUN_SUFFIX}_${objective}" \
          sbatch --parsable scripts/slurm/run_moving_objects_train.slurm
      )"
      printf '%s\t%s\t%s\t8\t%s\n' \
        "${run_name}" "${job_id}" "${latent_dim}" "${seed}" >> "${MANIFEST}"
    else
      printf 'DRY RUN: objective=%s z=%s levels=%s N=8 seed=%s run=%s\n' \
        "${objective}" "${latent_dim}" "${levels}" "${seed}" "${run_name}"
    fi
  done
done

if [[ "${SUBMIT:-0}" == "1" ]]; then
  printf 'Manifest: %s\n' "${MANIFEST}"
else
  printf 'Dry run only. Re-run with SUBMIT=1 to submit 12 hard-rate causal controls.\n'
fi
