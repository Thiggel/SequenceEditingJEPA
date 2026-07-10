#!/usr/bin/env bash
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p logs
source scripts/env.sh

MODELS=(cls64_r8 cls128_r8)
SEEDS=(1707 2707 3707)
STEP_COUNTS=(5000 15000 50000)
LEARNING_RATE="${LEARNING_RATE_OVERRIDE:-3.0e-4}"

if [[ -n "${MODELS_OVERRIDE:-}" ]]; then
  read -r -a MODELS <<< "${MODELS_OVERRIDE}"
fi
if [[ -n "${SEEDS_OVERRIDE:-}" ]]; then
  read -r -a SEEDS <<< "${SEEDS_OVERRIDE}"
fi
if [[ -n "${STEP_COUNTS_OVERRIDE:-}" ]]; then
  read -r -a STEP_COUNTS <<< "${STEP_COUNTS_OVERRIDE}"
fi

IDS=()
for model in "${MODELS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for steps in "${STEP_COUNTS[@]}"; do
      if (( steps <= 5000 )); then
        eval_every=1000
      elif (( steps <= 15000 )); then
        eval_every=2500
      else
        eval_every=5000
      fi
      suffix="v4_length_lr${LEARNING_RATE}_steps${steps}_seed${seed}"
      run_name="semantic_mix_${model}_ema_${suffix}"
      if [[ "${SUBMIT:-0}" == "1" ]]; then
        train_job="$(
          DATA_CONFIG=semantic_mix \
          MODEL_CONFIG="${model}" \
          OBJECTIVE_CONFIG=ema \
          SEED="${seed}" \
          LEARNING_RATE="${LEARNING_RATE}" \
          MAX_STEPS="${steps}" \
          EVAL_EVERY_STEPS="${eval_every}" \
          SAVE_EVERY_STEPS="${steps}" \
          EXTRA_HYDRA_OVERRIDES="eval.run_initial_probes=false eval.run_probes_during_training=false" \
          RUN_SUFFIX="${suffix}" \
            sbatch --parsable scripts/slurm/run_object_dynamics_train.slurm
        )"
        probe_job="$(
          RUN_NAME="${run_name}" \
          PROBE_OUTPUT_NAME=probe_eval_balanced_v4.json \
            sbatch --parsable --dependency="afterok:${train_job}" scripts/slurm/run_object_dynamics_probe_eval.slurm
        )"
        IDS+=("${run_name}:train=${train_job}:probe=${probe_job}")
      else
        printf 'DRY RUN: DATA_CONFIG=semantic_mix MODEL_CONFIG=%s OBJECTIVE_CONFIG=ema SEED=%s LEARNING_RATE=%s MAX_STEPS=%s EVAL_EVERY_STEPS=%s SAVE_EVERY_STEPS=%s EXTRA_HYDRA_OVERRIDES="eval.run_initial_probes=false eval.run_probes_during_training=false" RUN_SUFFIX=%s sbatch scripts/slurm/run_object_dynamics_train.slurm; RUN_NAME=%s PROBE_OUTPUT_NAME=probe_eval_balanced_v4.json sbatch --dependency=afterok:<train_job> scripts/slurm/run_object_dynamics_probe_eval.slurm\n' \
          "${model}" "${seed}" "${LEARNING_RATE}" "${steps}" "${eval_every}" "${steps}" "${suffix}" "${run_name}"
      fi
    done
  done
done

if [[ "${SUBMIT:-0}" == "1" ]]; then
  printf 'Submitted object-dynamics length calibration jobs:\n'
  printf '  %s\n' "${IDS[@]}"
else
  printf 'Dry run only. Re-run with SUBMIT=1 to submit.\n'
fi
