#!/usr/bin/env bash
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p logs

DATASETS=(semantic_mix)
MODELS=(cls64_r1 cls64_r8)
OBJECTIVES=(base)
LEARNING_RATES=(1.0e-4 3.0e-4 1.0e-3)
STEP_COUNTS=(500 1500)

IDS=()
for data in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    for objective in "${OBJECTIVES[@]}"; do
      for lr in "${LEARNING_RATES[@]}"; do
        for steps in "${STEP_COUNTS[@]}"; do
          suffix="prestage_lr${lr}_steps${steps}"
          if [[ "${SUBMIT:-0}" == "1" ]]; then
            job_id="$(
              DATA_CONFIG="${data}" \
              MODEL_CONFIG="${model}" \
              OBJECTIVE_CONFIG="${objective}" \
              LEARNING_RATE="${lr}" \
              MAX_STEPS="${steps}" \
              RUN_SUFFIX="${suffix}" \
                sbatch --parsable scripts/slurm/run_object_dynamics_train.slurm
            )"
            IDS+=("${data}/${model}/${objective}/${suffix}:${job_id}")
          else
            printf 'DRY RUN: DATA_CONFIG=%s MODEL_CONFIG=%s OBJECTIVE_CONFIG=%s LEARNING_RATE=%s MAX_STEPS=%s RUN_SUFFIX=%s sbatch scripts/slurm/run_object_dynamics_train.slurm\n' \
              "${data}" "${model}" "${objective}" "${lr}" "${steps}" "${suffix}"
          fi
        done
      done
    done
  done
done

if [[ "${SUBMIT:-0}" == "1" ]]; then
  printf 'Submitted object-dynamics prestage jobs:\n'
  printf '  %s\n' "${IDS[@]}"
else
  printf 'Dry run only. Re-run with SUBMIT=1 to submit.\n'
fi
