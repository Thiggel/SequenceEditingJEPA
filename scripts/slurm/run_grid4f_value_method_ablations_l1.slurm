#!/usr/bin/env bash
#SBATCH --job-name=puzzle_grid4f_value_methods
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-1

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

METHODS=(cvl_multipositive muzero_lite)
METHOD="${METHODS[$SLURM_ARRAY_TASK_ID]}"
RUN_NAME="sudoku_jepa_5m_value_method_l1_${METHOD}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/${RUN_NAME}"
SEED=$((801 + SLURM_ARRAY_TASK_ID))

COMMON_OVERRIDES=(
  "seed=${SEED}"
  "output_dir=${RUN_ROOT}"
  "model.hierarchy_levels=1"
  "training.hierarchy_weight=0.0"
  "+training.goal_energy_aux_batch_size=64"
  "+training.goal_energy_aux_regression_weight=1.0"
)

METHOD_OVERRIDES=()
case "$METHOD" in
  cvl_multipositive)
    METHOD_OVERRIDES=(
      "+training.goal_energy_contrastive_loss=infonce"
      "+training.goal_energy_contrastive_weight=1.0"
      "+training.goal_energy_contrastive_temperature=0.1"
      "+training.goal_energy_positive_mode=all_goal_correct"
      "+training.goal_energy_positive_count=8"
      "+training.goal_energy_negative_count=32"
    )
    ;;
  muzero_lite)
    METHOD_OVERRIDES=(
      "+model.use_action_policy_head=true"
      "+training.action_policy_weight=1.0"
      "+training.goal_energy_positive_count=1"
      "+training.goal_energy_negative_count=8"
    )
    ;;
  *)
    echo "Unknown method ${METHOD}" >&2
    exit 2
    ;;
esac

sleep $((SLURM_ARRAY_TASK_ID * 30))

python -m puzzle_jepa.train.grid0 \
  --config-name grid4a_sudoku_jepa_5m_goal_energy_cem_l1 \
  "${COMMON_OVERRIDES[@]}" \
  "${METHOD_OVERRIDES[@]}"

python -m puzzle_jepa.eval.diagnostics \
  --run-root "$RUN_ROOT" \
  --output-dir "$RUN_ROOT/diagnostics_reset_goal_energy" \
  --rank-examples 0 \
  --drift-examples 0 \
  --trace-examples 0 \
  --planning-examples 128 \
  --planning-score goal_energy \
  --planning-beam-size 4 \
  --planning-branch-size 8 \
  --reset-cadences 4 \
  --cem-examples 0 \
  --max-unroll-steps 81

python -m puzzle_jepa.eval.diagnostics \
  --run-root "$RUN_ROOT" \
  --output-dir "$RUN_ROOT/diagnostics_reset_oracle_calibration" \
  --rank-examples 0 \
  --drift-examples 0 \
  --trace-examples 0 \
  --planning-examples 128 \
  --planning-score latent_goal \
  --planning-beam-size 4 \
  --planning-branch-size 8 \
  --reset-cadences 4 \
  --cem-examples 0 \
  --max-unroll-steps 81
