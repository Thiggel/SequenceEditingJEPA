#!/usr/bin/env bash
#SBATCH --job-name=grid_goal_h1r_train
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-16%17
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

VARIANTS=(
  anchor_h1
  action_token
  action_local_feature
  action_old_local_value
  action_old_local_concat
  dynamics_affected
  dynamics_affected_context
  no_temporal
  no_progress
  no_action_rank
  no_terminal_corrupt
  no_vicreg
  minimal_aux
  hier_none
  hier_l4
  hier_l16
  hier_l4_l16_l32
)

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if (( INDEX < 0 || INDEX >= ${#VARIANTS[@]} )); then
  echo "Unknown SLURM_ARRAY_TASK_ID=${INDEX}; valid range is 0-$(( ${#VARIANTS[@]} - 1 ))" >&2
  exit 2
fi

VARIANT="${VARIANTS[$INDEX]}"
RUN_NAME="grid_goal_h1_recipe_${VARIANT}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/grid_goal_h1_recipe/${RUN_NAME}"

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
  "model.predict_delta=true"
  "model.dynamics_weighting=uniform"
  "model.affected_dynamics_weight=${AFFECTED_DYNAMICS_WEIGHT:-8.0}"
  "model.context_dynamics_weight=${CONTEXT_DYNAMICS_WEIGHT:-2.0}"
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
  "model.terminal_corrupt_weight=${TERMINAL_CORRUPT_WEIGHT:-1.0}"
  "model.policy_prior_weight=0.0"
  "model.policy_prior_planning_weight=0.0"
  "model.temporal_straightening_weight=${TEMPORAL_STRAIGHTENING_WEIGHT:-0.1}"
)

EXTRA_OVERRIDES=()
case "${VARIANT}" in
  anchor_h1)
    EXTRA_OVERRIDES=()
    ;;
  action_token)
    EXTRA_OVERRIDES=("model.action_conditioning=action_token")
    ;;
  action_local_feature)
    EXTRA_OVERRIDES=("model.action_conditioning=local_action_feature")
    ;;
  action_old_local_value)
    EXTRA_OVERRIDES=("model.action_conditioning=old_local_value")
    ;;
  action_old_local_concat)
    EXTRA_OVERRIDES=("model.action_conditioning=old_local_concat")
    ;;
  dynamics_affected)
    EXTRA_OVERRIDES=("model.dynamics_weighting=affected")
    ;;
  dynamics_affected_context)
    EXTRA_OVERRIDES=("model.dynamics_weighting=affected_context")
    ;;
  no_temporal)
    EXTRA_OVERRIDES=("model.temporal_straightening_weight=0.0")
    ;;
  no_progress)
    EXTRA_OVERRIDES=("model.progress_rank_target=none" "model.progress_rank_weight=0.0")
    ;;
  no_action_rank)
    EXTRA_OVERRIDES=("model.action_rank_mode=none" "model.action_rank_weight=0.0")
    ;;
  no_terminal_corrupt)
    EXTRA_OVERRIDES=("model.terminal_corrupt_weight=0.0")
    ;;
  no_vicreg)
    EXTRA_OVERRIDES=("model.regularizer=none")
    ;;
  minimal_aux)
    EXTRA_OVERRIDES=(
      "model.temporal_straightening_weight=0.0"
      "model.progress_rank_target=none"
      "model.progress_rank_weight=0.0"
      "model.action_rank_mode=none"
      "model.action_rank_weight=0.0"
      "model.terminal_corrupt_weight=0.0"
      "model.regularizer=none"
    )
    ;;
  hier_none)
    EXTRA_OVERRIDES=("model.hierarchy_levels=[]" "model.hierarchy_loss_weight=0.0")
    ;;
  hier_l4)
    EXTRA_OVERRIDES=("model.hierarchy_levels=[4]")
    ;;
  hier_l16)
    EXTRA_OVERRIDES=("model.hierarchy_levels=[16]")
    ;;
  hier_l4_l16_l32)
    EXTRA_OVERRIDES=("model.hierarchy_levels=[4,16,32]")
    ;;
  *)
    echo "Unknown H1 recipe variant ${VARIANT}" >&2
    exit 2
    ;;
esac

python -m puzzle_jepa.train.grid_goal_sudoku "${COMMON_OVERRIDES[@]}" "${EXTRA_OVERRIDES[@]}"
