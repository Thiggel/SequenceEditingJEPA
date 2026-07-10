#!/usr/bin/env bash
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p logs

MODELS=(cls64_r1 cls64_r8)
OBJECTIVES=(ema sigreg)
SEEDS=(2707 3707)
LEARNING_RATE="3.0e-4"
MAX_STEPS="5000"

IDS=()
for model in "${MODELS[@]}"; do
  for objective in "${OBJECTIVES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      suffix="stability_lr${LEARNING_RATE}_steps${MAX_STEPS}_seed${seed}"
      run_name="semantic_mix_${model}_${objective}_${suffix}"
      if [[ "${SUBMIT:-0}" == "1" ]]; then
        train_job="$(
          DATA_CONFIG=semantic_mix \
          MODEL_CONFIG="${model}" \
          OBJECTIVE_CONFIG="${objective}" \
          SEED="${seed}" \
          LEARNING_RATE="${LEARNING_RATE}" \
          MAX_STEPS="${MAX_STEPS}" \
          RUN_SUFFIX="${suffix}" \
            sbatch --parsable scripts/slurm/run_object_dynamics_train.slurm
        )"
        probe_job="$(
          RUN_NAME="${run_name}" \
            sbatch --parsable --dependency="afterok:${train_job}" scripts/slurm/run_object_dynamics_probe_eval.slurm
        )"
        IDS+=("${run_name}:train=${train_job}:probe=${probe_job}")
      else
        printf 'DRY RUN: DATA_CONFIG=semantic_mix MODEL_CONFIG=%s OBJECTIVE_CONFIG=%s SEED=%s LEARNING_RATE=%s MAX_STEPS=%s RUN_SUFFIX=%s sbatch scripts/slurm/run_object_dynamics_train.slurm; RUN_NAME=%s sbatch --dependency=afterok:<train_job> scripts/slurm/run_object_dynamics_probe_eval.slurm\n' \
          "${model}" "${objective}" "${seed}" "${LEARNING_RATE}" "${MAX_STEPS}" "${suffix}" "${run_name}"
      fi
    done
  done
done

if [[ "${SUBMIT:-0}" == "1" ]]; then
  printf 'Submitted object-dynamics stability replications:\n'
  printf '  %s\n' "${IDS[@]}"
else
  printf 'Dry run only. Re-run with SUBMIT=1 to submit.\n'
fi
