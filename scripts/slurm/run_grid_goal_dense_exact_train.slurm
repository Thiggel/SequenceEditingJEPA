#!/usr/bin/env bash
#SBATCH --job-name=grid_goal_dense_exact_train
#SBATCH --partition=rtxpro6k,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --array=0-2%3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

VARIANTS=(
  dense_exact_k8_uniform
  dense_exact_k8_inv_sqrt
  dense_exact_k8_gamma
)

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if (( INDEX < 0 || INDEX >= ${#VARIANTS[@]} )); then
  echo "Unknown SLURM_ARRAY_TASK_ID=${INDEX}; valid range is 0-$(( ${#VARIANTS[@]} - 1 ))" >&2
  exit 2
fi

VARIANT="${VARIANTS[$INDEX]}"
RUN_NAME="grid_goal_dense_exact_${VARIANT}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/grid_goal_dense_exact/${RUN_NAME}"

COMMON_OVERRIDES=(
  --config-name grid_goal_sudoku
  "seed=${SEED:-5204}"
  "ablation=R4_no_goal_nce"
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
  "model.goal_nce_weight=0.0"
  "model.goal_distance_field_weight=0.0"
  "model.goal_target_mode=target_stopgrad"
  "model.dense_future_weight=1.0"
  "model.dense_rollout_all_steps=false"
  "model.dense_rollout_variable_starts=true"
  "model.multi_step_horizons=[8]"
  "model.hierarchy_levels=[4,16]"
  "model.hierarchy_loss_weight=${HIERARCHY_LOSS_WEIGHT:-1.0}"
  "model.hierarchy_dense_future_weight=0.0"
  "model.shared_hierarchy_predictor=false"
  "model.goal_conditioning=context"
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

EXTRA_OVERRIDES=()
case "${VARIANT}" in
  dense_exact_k8_uniform)
    EXTRA_OVERRIDES=("model.dense_rollout_weighting=uniform")
    ;;
  dense_exact_k8_inv_sqrt)
    EXTRA_OVERRIDES=("model.dense_rollout_weighting=inverse_sqrt")
    ;;
  dense_exact_k8_gamma)
    EXTRA_OVERRIDES=(
      "model.dense_rollout_weighting=geometric"
      "model.dense_rollout_gamma=${DENSE_ROLLOUT_GAMMA:-0.8}"
    )
    ;;
  *)
    echo "Unknown dense exact variant ${VARIANT}" >&2
    exit 2
    ;;
esac

python -m puzzle_jepa.train.grid_goal_sudoku "${COMMON_OVERRIDES[@]}" "${EXTRA_OVERRIDES[@]}"
