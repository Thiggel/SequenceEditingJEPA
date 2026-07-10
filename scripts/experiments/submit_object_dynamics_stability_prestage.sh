#!/usr/bin/env bash
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p logs

DATASETS=(semantic_mix)
MODELS=(cls64_r1 cls64_r8)
OBJECTIVES=(ema vicreg sigreg)
read -r -a LEARNING_RATES <<< "${LEARNING_RATES_OVERRIDE:-1.0e-4 3.0e-4}"
read -r -a SEEDS <<< "${SEEDS_OVERRIDE:-1707}"
MAX_STEPS="${MAX_STEPS_OVERRIDE:-5000}"

IDS=()
for data in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    for objective in "${OBJECTIVES[@]}"; do
      for lr in "${LEARNING_RATES[@]}"; do
        for seed in "${SEEDS[@]}"; do
          suffix="stability_lr${lr}_steps${MAX_STEPS}_seed${seed}"
          if [[ "${SUBMIT:-0}" == "1" ]]; then
            job_id="$(
              DATA_CONFIG="${data}" \
              MODEL_CONFIG="${model}" \
              OBJECTIVE_CONFIG="${objective}" \
              SEED="${seed}" \
              LEARNING_RATE="${lr}" \
              MAX_STEPS="${MAX_STEPS}" \
              RUN_SUFFIX="${suffix}" \
                sbatch --parsable scripts/slurm/run_object_dynamics_train.slurm
            )"
            IDS+=("${data}/${model}/${objective}/${suffix}:${job_id}")
          else
            printf 'DRY RUN: DATA_CONFIG=%s MODEL_CONFIG=%s OBJECTIVE_CONFIG=%s SEED=%s LEARNING_RATE=%s MAX_STEPS=%s RUN_SUFFIX=%s sbatch scripts/slurm/run_object_dynamics_train.slurm\n' \
              "${data}" "${model}" "${objective}" "${seed}" "${lr}" "${MAX_STEPS}" "${suffix}"
          fi
        done
      done
    done
  done
done

if [[ "${SUBMIT:-0}" == "1" ]]; then
  printf 'Submitted object-dynamics stability prestage jobs:\n'
  printf '  %s\n' "${IDS[@]}"
else
  printf 'Dry run only. Re-run with SUBMIT=1 to submit.\n'
fi
