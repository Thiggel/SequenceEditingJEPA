#!/usr/bin/env bash
set -euo pipefail

printf 'Retired: this historical gate contains a full-grid latent row. Use submit_moving_objects_bottleneck.sh.\n' >&2
exit 2

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p logs
source scripts/env.sh

DATASETS=(object_blocked frontier_build interleaved_build global_random random_off_manifold)
SEEDS=(1707 2707 3707)
ROWS=(cls128_r8:ema cls128_r8:reconstruction grid128_r8:ema)
LEARNING_RATE="${LEARNING_RATE_OVERRIDE:-3.0e-4}"
MAX_STEPS="${MAX_STEPS_OVERRIDE:-5000}"

if [[ -n "${DATASETS_OVERRIDE:-}" ]]; then
  read -r -a DATASETS <<< "${DATASETS_OVERRIDE}"
fi
if [[ -n "${SEEDS_OVERRIDE:-}" ]]; then
  read -r -a SEEDS <<< "${SEEDS_OVERRIDE}"
fi
if [[ -n "${ROWS_OVERRIDE:-}" ]]; then
  read -r -a ROWS <<< "${ROWS_OVERRIDE}"
fi

IDS=()
for data in "${DATASETS[@]}"; do
  for row in "${ROWS[@]}"; do
    model="${row%%:*}"
    objective="${row##*:}"
    for seed in "${SEEDS[@]}"; do
      suffix="trajectory_gate_steps${MAX_STEPS}_seed${seed}"
      run_name="${data}_${model}_${objective}_${suffix}"
      if [[ "${SUBMIT:-0}" == "1" ]]; then
        train_job="$(
          DATA_CONFIG="${data}" \
          MODEL_CONFIG="${model}" \
          OBJECTIVE_CONFIG="${objective}" \
          SEED="${seed}" \
          LEARNING_RATE="${LEARNING_RATE}" \
          MAX_STEPS="${MAX_STEPS}" \
          EVAL_EVERY_STEPS="${MAX_STEPS}" \
          SAVE_EVERY_STEPS="${MAX_STEPS}" \
          EXTRA_HYDRA_OVERRIDES="eval.run_initial_probes=false eval.run_probes_during_training=false" \
          RUN_SUFFIX="${suffix}" \
            sbatch --parsable scripts/slurm/run_object_dynamics_train.slurm
        )"
        common_probe="$(
          RUN_NAME="${run_name}" \
          PROBE_OUTPUT_NAME=probe_eval_common_v4.json \
          PROBE_TRAJECTORY_KIND=semantic_mix \
            sbatch --parsable --dependency="afterok:${train_job}" scripts/slurm/run_object_dynamics_probe_eval.slurm
        )"
        in_domain_probe="$(
          RUN_NAME="${run_name}" \
          PROBE_OUTPUT_NAME=probe_eval_in_domain_v4.json \
          PROBE_TRAJECTORY_KIND="${data}" \
            sbatch --parsable --dependency="afterok:${train_job}" scripts/slurm/run_object_dynamics_probe_eval.slurm
        )"
        IDS+=("${run_name}:train=${train_job}:common=${common_probe}:in_domain=${in_domain_probe}")
      else
        printf 'DRY RUN: %s/%s/%s seed=%s steps=%s; common=semantic_mix; in_domain=%s\n' \
          "${data}" "${model}" "${objective}" "${seed}" "${MAX_STEPS}" "${data}"
      fi
    done
  done
done

if [[ "${SUBMIT:-0}" == "1" ]]; then
  printf 'Submitted object-dynamics trajectory gate jobs:\n'
  printf '  %s\n' "${IDS[@]}"
else
  printf 'Dry run only. Re-run with SUBMIT=1 to submit.\n'
fi
