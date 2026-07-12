#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs
source scripts/env.sh

ROWS=(
  "2 0"
  "2 2"
  "2 4"
  "2 16"
  "4 0"
  "4 2"
  "4 4"
  "4 16"
  "8 0"
  "8 2"
  "8 4"
  "8 16"
)
OBJECT_COUNTS=(2 4 8)
SEEDS=(1707 2707 3707)
RUN_SUFFIX="${RUN_SUFFIX:-rate_bottleneck_v1_steps${MAX_STEPS:-5000}}"
MANIFEST="${PUZZLE_JEPA_WORK_ROOT}/runs/moving_objects/manifests/${RUN_SUFFIX}.tsv"
mkdir -p "$(dirname "${MANIFEST}")"

declare -A EXISTING_RUNS=()
if [[ "${SUBMIT:-0}" == "1" && "${RESUME:-0}" == "1" && -f "${MANIFEST}" ]]; then
  while IFS=$'\t' read -r run_name _; do
    if [[ "${run_name}" != "run_name" ]]; then
      EXISTING_RUNS["${run_name}"]=1
    fi
  done < "${MANIFEST}"
elif [[ "${SUBMIT:-0}" == "1" ]]; then
  printf 'run_name\tjob_id\tlatent_dim\tmax_objects\tseed\n' > "${MANIFEST}"
fi
dependency_args=()
if [[ -n "${DEPENDENCY:-}" ]]; then
  dependency_args=(--dependency="${DEPENDENCY}")
fi
for row in "${ROWS[@]}"; do
  read -r latent_dim levels <<<"${row}"
  case "${levels}" in
    0) bits_per_dimension=0 ;;
    2) bits_per_dimension=1 ;;
    4) bits_per_dimension=2 ;;
    16) bits_per_dimension=4 ;;
    *) printf 'Unsupported quantization level: %s\n' "${levels}" >&2; exit 2 ;;
  esac
  capacity_bits=$((latent_dim * bits_per_dimension))
  if [[ "${levels}" == "0" ]]; then
    capacity_bits=continuous
  fi
  for object_count in "${OBJECT_COUNTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      run_name="motion_n${object_count}_z${latent_dim}_q${levels}_${RUN_SUFFIX}_seed${seed}"
      if [[ "${SUBMIT:-0}" == "1" ]]; then
        if [[ -n "${EXISTING_RUNS[${run_name}]+x}" ]]; then
          continue
        fi
        job_id="$(
          LATENT_DIM="${latent_dim}" \
          LATENT_QUANTIZATION_LEVELS="${levels}" \
          MIN_OBJECTS="${object_count}" MAX_OBJECTS="${object_count}" \
          SEED="${seed}" DATA_CONFIG=reflected_motion OBJECTIVE_CONFIG=ema_vicreg_strong \
          MAX_STEPS="${MAX_STEPS:-5000}" LEARNING_RATE="${LEARNING_RATE:-3.0e-4}" \
          RUN_SUFFIX="q${levels}_${RUN_SUFFIX}" \
            sbatch --parsable "${dependency_args[@]}" scripts/slurm/run_moving_objects_train.slurm
        )"
        printf '%s\t%s\t%s\t%s\t%s\n' \
          "${run_name}" "${job_id}" "${latent_dim}" "${object_count}" "${seed}" \
          >> "${MANIFEST}"
      else
        printf 'DRY RUN: z=%s levels=%s capacity_bits=%s N=%s seed=%s run=%s\n' \
          "${latent_dim}" "${levels}" "${capacity_bits}" "${object_count}" \
          "${seed}" "${run_name}"
      fi
    done
  done
done

if [[ "${SUBMIT:-0}" == "1" ]]; then
  printf 'Manifest: %s\n' "${MANIFEST}"
else
  printf 'Dry run only. Re-run with SUBMIT=1 to submit 108 rate-constrained exact-load single-CLS jobs.\n'
fi
