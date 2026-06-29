#!/usr/bin/env bash
#SBATCH --job-name=grid_goal_h1x_train
#SBATCH --partition=rtxpro6k,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-11%12
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

VARIANTS=(
  rank_oracle_progress
  rank_both_progress
  rank_no_progress
  rank_pairwise_oracle_action
  rank_pairwise_both_action
  rank_listwise_pred_action
  rank_listwise_both_action
  rank_no_action
  hier_l4
  hier_l4_l16_l32
  hier_l4_l16_shared
  hier_l4_l16_hier_dense
)

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if (( INDEX < 0 || INDEX >= ${#VARIANTS[@]} )); then
  echo "Unknown SLURM_ARRAY_TASK_ID=${INDEX}; valid range is 0-$(( ${#VARIANTS[@]} - 1 ))" >&2
  exit 2
fi

VARIANT="${VARIANTS[$INDEX]}"
RUN_NAME="grid_goal_h1_extra_${VARIANT}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/grid_goal_h1_extra/${RUN_NAME}"

COMMON_OVERRIDES=(
  --config-name grid_goal_sudoku
  "seed=${SEED:-5204}"
  "ablation=R4_no_goal_nce"
  "output_dir=${RUN_ROOT}"
  "task.train_limit=${TRAIN_LIMIT:-null}"
  "task.eval_limit=${EVAL_LIMIT:-null}"
  "training.max_steps=${TRAIN_MAX_STEPS:-45000}"
  "training.batch_size=${BATCH_SIZE:-8}"
  "training.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS:-1}"
  "training.eval_every_steps=${EVAL_EVERY_STEPS:-1000}"
  "training.save_every_steps=${SAVE_EVERY_STEPS:-5000}"
  "training.learning_rate=${LEARNING_RATE:-1.0e-4}"
  "model.action_conditioning=affected_marker"
  "model.predict_delta=${PREDICT_DELTA:-false}"
  "model.dynamics_weighting=uniform"
  "model.regularizer=vicreg"
  "model.use_ema_target_encoder=true"
  "model.ema_decay=${EMA_DECAY:-0.995}"
  "model.goal_nce_weight=0.0"
  "model.dense_future_weight=${DENSE_FUTURE_WEIGHT:-1.0}"
  "model.multi_step_horizons=[1,4,8,16]"
  "model.hierarchy_levels=[4,16]"
  "model.hierarchy_loss_weight=${HIERARCHY_LOSS_WEIGHT:-1.0}"
  "model.hierarchy_dense_future_weight=0.0"
  "model.shared_hierarchy_predictor=false"
  "model.goal_conditioning=context"
  "model.progress_rank_target=predicted"
  "model.progress_rank_weight=${PROGRESS_RANK_WEIGHT:-1.0}"
  "model.action_rank_mode=pairwise"
  "model.action_rank_target=predicted"
  "model.action_rank_weight=${ACTION_RANK_WEIGHT:-1.0}"
  "model.policy_prior_weight=0.0"
  "model.policy_prior_planning_weight=0.0"
  "model.temporal_straightening_weight=${TEMPORAL_STRAIGHTENING_WEIGHT:-0.1}"
)

EXTRA_OVERRIDES=()
case "${VARIANT}" in
  rank_oracle_progress)
    EXTRA_OVERRIDES=("model.progress_rank_target=oracle")
    ;;
  rank_both_progress)
    EXTRA_OVERRIDES=("model.progress_rank_target=both")
    ;;
  rank_no_progress)
    EXTRA_OVERRIDES=("model.progress_rank_target=none" "model.progress_rank_weight=0.0")
    ;;
  rank_pairwise_oracle_action)
    EXTRA_OVERRIDES=("model.action_rank_mode=pairwise" "model.action_rank_target=oracle")
    ;;
  rank_pairwise_both_action)
    EXTRA_OVERRIDES=("model.action_rank_mode=pairwise" "model.action_rank_target=both")
    ;;
  rank_listwise_pred_action)
    EXTRA_OVERRIDES=(
      "model.action_rank_mode=listwise"
      "model.action_rank_target=predicted"
      "model.listwise_action_rank_max_actions=${LISTWISE_ACTION_RANK_MAX_ACTIONS:-128}"
    )
    ;;
  rank_listwise_both_action)
    EXTRA_OVERRIDES=(
      "model.action_rank_mode=listwise"
      "model.action_rank_target=both"
      "model.listwise_action_rank_max_actions=${LISTWISE_ACTION_RANK_MAX_ACTIONS:-128}"
    )
    ;;
  rank_no_action)
    EXTRA_OVERRIDES=("model.action_rank_mode=none" "model.action_rank_weight=0.0")
    ;;
  hier_l4)
    EXTRA_OVERRIDES=("model.hierarchy_levels=[4]")
    ;;
  hier_l4_l16_l32)
    EXTRA_OVERRIDES=("model.hierarchy_levels=[4,16,32]")
    ;;
  hier_l4_l16_shared)
    EXTRA_OVERRIDES=("model.shared_hierarchy_predictor=true")
    ;;
  hier_l4_l16_hier_dense)
    EXTRA_OVERRIDES=("model.hierarchy_dense_future_weight=${HIERARCHY_DENSE_FUTURE_WEIGHT:-1.0}")
    ;;
  *)
    echo "Unknown H1 extra variant ${VARIANT}" >&2
    exit 2
    ;;
esac

python -m puzzle_jepa.train.grid_goal_sudoku "${COMMON_OVERRIDES[@]}" "${EXTRA_OVERRIDES[@]}"
