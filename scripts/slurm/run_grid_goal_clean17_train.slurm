#!/usr/bin/env bash
#SBATCH --job-name=grid_goal_clean17_train
#SBATCH --partition=rtxpro6k,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

VARIANTS=(
  W_uniform_H0_G_none
  W_uniform_H4_G_none
  W_uniform_H4_16_G_none
  W_uniform_H4_16_32_G_none
  W_inv_sqrt_H0_G_none
  W_inv_sqrt_H4_G_none
  W_inv_sqrt_H4_16_G_none
  W_inv_sqrt_H4_16_32_G_none
  W_gamma_H0_G_none
  W_gamma_H4_G_none
  W_gamma_H4_16_G_none
  W_gamma_H4_16_32_G_none
  G_ic_detached
  G_ic_non_detached
  G_ic_online_no_stopgrad
  G_ic_field_plus_mse
  G_ic_field_only
)

if [[ -n "${VARIANT:-}" ]]; then
  SELECTED_VARIANT="${VARIANT}"
else
  INDEX="${SLURM_ARRAY_TASK_ID:-0}"
  if (( INDEX < 0 || INDEX >= ${#VARIANTS[@]} )); then
    echo "Unknown SLURM_ARRAY_TASK_ID=${INDEX}; valid range is 0-$(( ${#VARIANTS[@]} - 1 ))" >&2
    exit 2
  fi
  SELECTED_VARIANT="${VARIANTS[$INDEX]}"
fi

RUN_NAME="grid_goal_clean17_${SELECTED_VARIANT}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/grid_goal_clean17/${RUN_NAME}"

COMMON_OVERRIDES=(
  --config-name grid_goal_sudoku
  "seed=${SEED:-5204}"
  "ablation=M0_full"
  "+experiment_variant=${SELECTED_VARIANT}"
  "output_dir=${RUN_ROOT}"
  "task.train_limit=${TRAIN_LIMIT:-null}"
  "task.eval_limit=${EVAL_LIMIT:-null}"
  "training.max_steps=${TRAIN_MAX_STEPS:-5000}"
  "training.batch_size=${BATCH_SIZE:-8}"
  "training.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS:-1}"
  "training.eval_every_steps=${EVAL_EVERY_STEPS:-500}"
  "training.save_every_steps=${SAVE_EVERY_STEPS:-5000}"
  "training.learning_rate=${LEARNING_RATE:-1.0e-4}"
  "training.warmup_steps=${WARMUP_STEPS:-500}"
  "model.action_conditioning=affected_marker"
  "model.predict_delta=true"
  "model.dynamics_weighting=uniform"
  "model.affected_dynamics_weight=${AFFECTED_DYNAMICS_WEIGHT:-8.0}"
  "model.context_dynamics_weight=${CONTEXT_DYNAMICS_WEIGHT:-2.0}"
  "model.regularizer=none"
  "model.use_ema_target_encoder=true"
  "model.ema_decay=${EMA_DECAY:-0.995}"
  "model.goal_mse_weight=0.0"
  "model.goal_nce_weight=0.0"
  "model.goal_distance_field_weight=0.0"
  "model.goal_target_mode=target_stopgrad"
  "model.goal_conditioning=context"
  "model.goal_conditioning_detach_state=false"
  "model.dense_future_weight=1.0"
  "model.dense_rollout_all_steps=false"
  "model.dense_rollout_variable_starts=true"
  "model.multi_step_horizons=[8]"
  "model.hierarchy_levels=[]"
  "model.hierarchy_loss_weight=0.0"
  "model.hierarchy_dense_future_weight=0.0"
  "model.shared_hierarchy_predictor=false"
  "model.progress_rank_target=none"
  "model.progress_rank_weight=0.0"
  "model.action_rank_mode=none"
  "model.action_rank_target=predicted"
  "model.action_rank_weight=0.0"
  "model.terminal_corrupt_weight=0.0"
  "model.policy_prior_weight=0.0"
  "model.policy_prior_planning_weight=0.0"
  "model.temporal_straightening_weight=0.0"
)

WEIGHT_OVERRIDES=()
HIERARCHY_OVERRIDES=()
GOAL_OVERRIDES=()
case "${SELECTED_VARIANT}" in
  W_uniform_*)
    WEIGHT_OVERRIDES=("model.dense_rollout_weighting=uniform")
    ;;
  W_inv_sqrt_*|G_*)
    WEIGHT_OVERRIDES=("model.dense_rollout_weighting=inverse_sqrt")
    ;;
  W_gamma_*)
    WEIGHT_OVERRIDES=(
      "model.dense_rollout_weighting=geometric"
      "model.dense_rollout_gamma=${DENSE_ROLLOUT_GAMMA:-0.8}"
    )
    ;;
  *)
    echo "Unknown weighting in variant ${SELECTED_VARIANT}" >&2
    exit 2
    ;;
esac

case "${SELECTED_VARIANT}" in
  *_H0_*)
    HIERARCHY_OVERRIDES=("model.hierarchy_levels=[]" "model.hierarchy_loss_weight=0.0")
    ;;
  *_H4_16_32_*)
    HIERARCHY_OVERRIDES=("model.hierarchy_levels=[4,16,32]" "model.hierarchy_loss_weight=1.0")
    ;;
  *_H4_16_*)
    HIERARCHY_OVERRIDES=("model.hierarchy_levels=[4,16]" "model.hierarchy_loss_weight=1.0")
    ;;
  *_H4_*)
    HIERARCHY_OVERRIDES=("model.hierarchy_levels=[4]" "model.hierarchy_loss_weight=1.0")
    ;;
  G_*)
    HIERARCHY_OVERRIDES=("model.hierarchy_levels=[4,16]" "model.hierarchy_loss_weight=1.0")
    ;;
  *)
    echo "Unknown hierarchy in variant ${SELECTED_VARIANT}" >&2
    exit 2
    ;;
esac

case "${SELECTED_VARIANT}" in
  W_*_G_none)
    GOAL_OVERRIDES=(
      "model.goal_mse_weight=0.0"
      "model.goal_conditioning=context"
      "model.goal_conditioning_detach_state=false"
    )
    ;;
  G_ic_detached)
    GOAL_OVERRIDES=(
      "model.goal_mse_weight=1.0"
      "model.goal_conditioning=initial_current"
      "model.goal_conditioning_detach_state=true"
      "model.goal_target_mode=target_stopgrad"
      "model.goal_distance_field_weight=0.0"
    )
    ;;
  G_ic_non_detached)
    GOAL_OVERRIDES=(
      "model.goal_mse_weight=1.0"
      "model.goal_conditioning=initial_current"
      "model.goal_conditioning_detach_state=false"
      "model.goal_target_mode=target_stopgrad"
      "model.goal_distance_field_weight=0.0"
    )
    ;;
  G_ic_online_no_stopgrad)
    GOAL_OVERRIDES=(
      "model.goal_mse_weight=1.0"
      "model.goal_conditioning=initial_current"
      "model.goal_conditioning_detach_state=false"
      "model.goal_target_mode=online_no_stopgrad"
      "model.goal_distance_field_weight=0.0"
    )
    ;;
  G_ic_field_plus_mse)
    GOAL_OVERRIDES=(
      "model.goal_mse_weight=1.0"
      "model.goal_conditioning=initial_current"
      "model.goal_conditioning_detach_state=true"
      "model.goal_target_mode=target_stopgrad"
      "model.goal_distance_field_weight=1.0"
    )
    ;;
  G_ic_field_only)
    GOAL_OVERRIDES=(
      "model.goal_mse_weight=0.0"
      "model.goal_conditioning=initial_current"
      "model.goal_conditioning_detach_state=true"
      "model.goal_target_mode=target_stopgrad"
      "model.goal_distance_field_weight=1.0"
    )
    ;;
  *)
    echo "Unknown goal mode in variant ${SELECTED_VARIANT}" >&2
    exit 2
    ;;
esac

python -m puzzle_jepa.train.grid_goal_sudoku \
  "${COMMON_OVERRIDES[@]}" \
  "${WEIGHT_OVERRIDES[@]}" \
  "${HIERARCHY_OVERRIDES[@]}" \
  "${GOAL_OVERRIDES[@]}"
