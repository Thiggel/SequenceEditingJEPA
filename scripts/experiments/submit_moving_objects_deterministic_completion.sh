#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs
source scripts/env.sh

LATENT_DIMS=(2 4 8 16 32 64)
MAX_OBJECT_COUNTS=(1 2 4 6 8)
SEEDS=(1707 2707 3707)
REUSED_CELLS=("4 4" "4 8" "32 4" "32 8")
RUN_SUFFIX="${RUN_SUFFIX:-deterministic_reflected_matrix_v2_steps${MAX_STEPS:-5000}}"
MANIFEST="${PUZZLE_JEPA_WORK_ROOT}/runs/moving_objects/manifests/${RUN_SUFFIX}.tsv"
CONFIRMATION_MANIFEST="${CONFIRMATION_MANIFEST:-${PUZZLE_JEPA_WORK_ROOT}/runs/moving_objects/manifests/deterministic_confirmation_v1_steps5000.tsv}"
mkdir -p "$(dirname "${MANIFEST}")"

is_reused_cell() {
  local candidate="$1 $2"
  local cell
  for cell in "${REUSED_CELLS[@]}"; do
    [[ "${candidate}" == "${cell}" ]] && return 0
  done
  return 1
}

if [[ "${SUBMIT:-0}" == "1" ]]; then
  [[ -s "${CONFIRMATION_MANIFEST}" ]] || {
    printf 'Missing confirmation manifest: %s\n' "${CONFIRMATION_MANIFEST}" >&2
    exit 1
  }
  printf 'run_name\tjob_id\tlatent_dim\tmax_objects\tseed\n' > "${MANIFEST}"
fi

new_count=0
reused_count=0
for latent_dim in "${LATENT_DIMS[@]}"; do
  for max_objects in "${MAX_OBJECT_COUNTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      if is_reused_cell "${latent_dim}" "${max_objects}"; then
        existing_name="motion_n${max_objects}_z${latent_dim}_deterministic_confirmation_v1_steps5000_reflected_motion_ema_vicreg_seed${seed}"
        if [[ "${SUBMIT:-0}" == "1" ]]; then
          existing_row="$(awk -F '\t' -v name="${existing_name}" '$1 == name {print; exit}' "${CONFIRMATION_MANIFEST}")"
          [[ -n "${existing_row}" ]] || {
            printf 'Missing reusable row: %s\n' "${existing_name}" >&2
            exit 1
          }
          printf '%s\n' "${existing_row}" >> "${MANIFEST}"
        else
          printf 'REUSE: z=%s N=%s seed=%s run=%s\n' \
            "${latent_dim}" "${max_objects}" "${seed}" "${existing_name}"
        fi
        reused_count=$((reused_count + 1))
        continue
      fi

      run_name="motion_n${max_objects}_z${latent_dim}_${RUN_SUFFIX}_seed${seed}"
      if [[ "${SUBMIT:-0}" == "1" ]]; then
        job_id="$(
          LATENT_DIM="${latent_dim}" MAX_OBJECTS="${max_objects}" SEED="${seed}" \
          DATA_CONFIG=reflected_motion OBJECTIVE_CONFIG=ema_vicreg \
          MAX_STEPS="${MAX_STEPS:-5000}" LEARNING_RATE="${LEARNING_RATE:-3.0e-4}" \
          RUN_SUFFIX="${RUN_SUFFIX}" \
            sbatch --parsable scripts/slurm/run_moving_objects_train.slurm
        )"
        printf '%s\t%s\t%s\t%s\t%s\n' \
          "${run_name}" "${job_id}" "${latent_dim}" "${max_objects}" "${seed}" >> "${MANIFEST}"
      else
        printf 'DRY RUN: z=%s N=%s seed=%s run=%s\n' \
          "${latent_dim}" "${max_objects}" "${seed}" "${run_name}"
      fi
      new_count=$((new_count + 1))
    done
  done
done

if [[ "${SUBMIT:-0}" == "1" ]]; then
  printf 'Manifest: %s\n' "${MANIFEST}"
fi
printf 'Rows: 90 deterministic single-CLS runs (%s new, %s reused).\n' \
  "${new_count}" "${reused_count}"
