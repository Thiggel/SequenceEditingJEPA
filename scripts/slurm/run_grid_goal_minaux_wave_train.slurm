#!/usr/bin/env bash
#SBATCH --job-name=grid_goal_minaux_train
#SBATCH --partition=rtxpro6k,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --array=0-28%29
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

VARIANTS=(
  base
  reg_vicreg
  reg_sigreg
  reg_no_ema
  reg_vicreg_no_ema
  reg_sigreg_no_ema
  reg_vicreg_sigreg
  reg_vicreg_sigreg_no_ema
  rank_pairwise_pred_action
  rank_listwise_pred_action
  rank_pairwise_oracle_action
  rank_listwise_oracle_action
  geom_temporal
  geom_pred_progress
  geom_oracle_progress
  dense_k1
  dense_k2
  dense_k4
  dense_k8
  dense_k16
  hier_none
  hier_l4
  hier_l16
  hier_l4_l16
  hier_l4_l16_l32
  goal_initial_current
  goal_no_stopgrad
  goal_initial_current_no_stopgrad
  goal_distance_field_distill
)

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if (( INDEX < 0 || INDEX >= ${#VARIANTS[@]} )); then
  echo "Unknown SLURM_ARRAY_TASK_ID=${INDEX}; valid range is 0-$(( ${#VARIANTS[@]} - 1 ))" >&2
  exit 2
fi

VARIANT="${VARIANTS[$INDEX]}"
RUN_NAME="grid_goal_minaux_wave_${VARIANT}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/grid_goal_minaux_wave/${RUN_NAME}"

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
  "model.dense_future_weight=${DENSE_FUTURE_WEIGHT:-1.0}"
  "model.dense_rollout_all_steps=false"
  "model.multi_step_horizons=[1,4,8,16]"
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
  base)
    EXTRA_OVERRIDES=()
    ;;
  reg_vicreg)
    EXTRA_OVERRIDES=("model.regularizer=vicreg")
    ;;
  reg_sigreg)
    EXTRA_OVERRIDES=("model.regularizer=sigreg")
    ;;
  reg_no_ema)
    EXTRA_OVERRIDES=("model.use_ema_target_encoder=false")
    ;;
  reg_vicreg_no_ema)
    EXTRA_OVERRIDES=("model.regularizer=vicreg" "model.use_ema_target_encoder=false")
    ;;
  reg_sigreg_no_ema)
    EXTRA_OVERRIDES=("model.regularizer=sigreg" "model.use_ema_target_encoder=false")
    ;;
  reg_vicreg_sigreg)
    EXTRA_OVERRIDES=("model.regularizer=both")
    ;;
  reg_vicreg_sigreg_no_ema)
    EXTRA_OVERRIDES=("model.regularizer=both" "model.use_ema_target_encoder=false")
    ;;
  rank_pairwise_pred_action)
    EXTRA_OVERRIDES=("model.action_rank_mode=pairwise" "model.action_rank_target=predicted" "model.action_rank_weight=1.0")
    ;;
  rank_listwise_pred_action)
    EXTRA_OVERRIDES=(
      "model.action_rank_mode=listwise"
      "model.action_rank_target=predicted"
      "model.action_rank_weight=1.0"
      "model.listwise_action_rank_max_actions=${LISTWISE_ACTION_RANK_MAX_ACTIONS:-128}"
    )
    ;;
  rank_pairwise_oracle_action)
    EXTRA_OVERRIDES=("model.action_rank_mode=pairwise" "model.action_rank_target=oracle" "model.action_rank_weight=1.0")
    ;;
  rank_listwise_oracle_action)
    EXTRA_OVERRIDES=(
      "model.action_rank_mode=listwise"
      "model.action_rank_target=oracle"
      "model.action_rank_weight=1.0"
      "model.listwise_action_rank_max_actions=${LISTWISE_ACTION_RANK_MAX_ACTIONS:-128}"
    )
    ;;
  geom_temporal)
    EXTRA_OVERRIDES=("model.temporal_straightening_weight=0.1")
    ;;
  geom_pred_progress)
    EXTRA_OVERRIDES=("model.progress_rank_target=predicted" "model.progress_rank_weight=1.0")
    ;;
  geom_oracle_progress)
    EXTRA_OVERRIDES=("model.progress_rank_target=oracle" "model.progress_rank_weight=1.0")
    ;;
  dense_k1)
    EXTRA_OVERRIDES=("model.dense_rollout_all_steps=true" "model.multi_step_horizons=[1]")
    ;;
  dense_k2)
    EXTRA_OVERRIDES=("model.dense_rollout_all_steps=true" "model.multi_step_horizons=[2]")
    ;;
  dense_k4)
    EXTRA_OVERRIDES=("model.dense_rollout_all_steps=true" "model.multi_step_horizons=[4]")
    ;;
  dense_k8)
    EXTRA_OVERRIDES=("model.dense_rollout_all_steps=true" "model.multi_step_horizons=[8]")
    ;;
  dense_k16)
    EXTRA_OVERRIDES=("model.dense_rollout_all_steps=true" "model.multi_step_horizons=[16]")
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
  hier_l4_l16)
    EXTRA_OVERRIDES=("model.hierarchy_levels=[4,16]")
    ;;
  hier_l4_l16_l32)
    EXTRA_OVERRIDES=("model.hierarchy_levels=[4,16,32]")
    ;;
  goal_initial_current)
    EXTRA_OVERRIDES=("model.goal_conditioning=initial_current")
    ;;
  goal_no_stopgrad)
    EXTRA_OVERRIDES=("model.goal_target_mode=online_no_stopgrad")
    ;;
  goal_initial_current_no_stopgrad)
    EXTRA_OVERRIDES=("model.goal_conditioning=initial_current" "model.goal_target_mode=online_no_stopgrad")
    ;;
  goal_distance_field_distill)
    EXTRA_OVERRIDES=("model.goal_distance_field_weight=1.0")
    ;;
  *)
    echo "Unknown minaux variant ${VARIANT}" >&2
    exit 2
    ;;
esac

python -m puzzle_jepa.train.grid_goal_sudoku "${COMMON_OVERRIDES[@]}" "${EXTRA_OVERRIDES[@]}"
