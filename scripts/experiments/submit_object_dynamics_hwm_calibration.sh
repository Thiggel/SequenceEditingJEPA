#!/usr/bin/env bash
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p logs
source scripts/env.sh

SEEDS=(1707)
MACRO_DIMS=(4 8 16)
LEARNING_RATE="${LEARNING_RATE_OVERRIDE:-3.0e-4}"
MAX_STEPS="${MAX_STEPS_OVERRIDE:-5000}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS_OVERRIDE:-1000}"

if [[ -n "${SEEDS_OVERRIDE:-}" ]]; then
  read -r -a SEEDS <<< "${SEEDS_OVERRIDE}"
fi
if [[ -n "${MACRO_DIMS_OVERRIDE:-}" ]]; then
  read -r -a MACRO_DIMS <<< "${MACRO_DIMS_OVERRIDE}"
fi

IDS=()
for seed in "${SEEDS[@]}"; do
  low_suffix="hwm_calibration_low_seed${seed}"
  low_run="semantic_mix_cls128_r8_base_${low_suffix}"
  output_root="${PUZZLE_JEPA_WORK_ROOT:-${WORK:-/tmp}/sequence-editing-object-dynamics-20260708}"
  low_checkpoint="${output_root}/runs/object_dynamics/${low_run}/checkpoint.pt"
  if [[ "${SUBMIT:-0}" == "1" ]]; then
    low_job="$(
      DATA_CONFIG=semantic_mix \
      MODEL_CONFIG=cls128_r8 \
      OBJECTIVE_CONFIG=base \
      SEED="${seed}" \
      LEARNING_RATE="${LEARNING_RATE}" \
      MAX_STEPS="${MAX_STEPS}" \
      EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS}" \
      SAVE_EVERY_STEPS="${MAX_STEPS}" \
      EXTRA_HYDRA_OVERRIDES="eval.run_initial_probes=false eval.run_probes_during_training=false" \
      RUN_SUFFIX="${low_suffix}" \
        sbatch --parsable scripts/slurm/run_object_dynamics_train.slurm
    )"
    low_probe="$(
      RUN_NAME="${low_run}" PROBE_OUTPUT_NAME=probe_eval_balanced_v4.json \
        sbatch --parsable --dependency="afterok:${low_job}" scripts/slurm/run_object_dynamics_probe_eval.slurm
    )"
    IDS+=("${low_run}:train=${low_job}:probe=${low_probe}")
  else
    low_job="<low_job>"
    printf 'DRY RUN: low %s seed=%s steps=%s; probe afterok\n' "${low_run}" "${seed}" "${MAX_STEPS}"
  fi

  for macro_dim in "${MACRO_DIMS[@]}"; do
    for schedule in joint staged; do
      suffix="hwm_calibration_${schedule}_d${macro_dim}_seed${seed}"
      run_name="semantic_mix_h_cls128_h8_base_${suffix}"
      dependency_args=()
      initial_checkpoint="null"
      trainable_components="all"
      if [[ "${schedule}" == "staged" ]]; then
        dependency_args=(--dependency="afterok:${low_job}")
        initial_checkpoint="${low_checkpoint}"
        trainable_components="hierarchy_only"
      fi
      if [[ "${SUBMIT:-0}" == "1" ]]; then
        train_job="$(
          DATA_CONFIG=semantic_mix \
          MODEL_CONFIG=h_cls128_h8 \
          OBJECTIVE_CONFIG=base \
          SEED="${seed}" \
          LEARNING_RATE="${LEARNING_RATE}" \
          MAX_STEPS="${MAX_STEPS}" \
          EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS}" \
          SAVE_EVERY_STEPS="${MAX_STEPS}" \
          INITIAL_CHECKPOINT="${initial_checkpoint}" \
          TRAINABLE_COMPONENTS="${trainable_components}" \
          EXTRA_HYDRA_OVERRIDES="model.macro_action_dim=${macro_dim} eval.run_initial_probes=false eval.run_probes_during_training=false" \
          RUN_SUFFIX="${suffix}" \
            sbatch --parsable "${dependency_args[@]}" scripts/slurm/run_object_dynamics_train.slurm
        )"
        probe_job="$(
          RUN_NAME="${run_name}" PROBE_OUTPUT_NAME=probe_eval_balanced_v4.json \
            sbatch --parsable --dependency="afterok:${train_job}" scripts/slurm/run_object_dynamics_probe_eval.slurm
        )"
        IDS+=("${run_name}:train=${train_job}:probe=${probe_job}")
      else
        printf 'DRY RUN: %s macro_dim=%s seed=%s steps=%s initial=%s trainable=%s dependency=%s; probe afterok\n' \
          "${run_name}" "${macro_dim}" "${seed}" "${MAX_STEPS}" "${initial_checkpoint}" \
          "${trainable_components}" "${dependency_args[*]:-none}"
      fi
    done
  done
done

if [[ "${SUBMIT:-0}" == "1" ]]; then
  printf 'Submitted object-dynamics HWM calibration jobs:\n'
  printf '  %s\n' "${IDS[@]}"
else
  printf 'Dry run only. Re-run with SUBMIT=1 to submit.\n'
fi
