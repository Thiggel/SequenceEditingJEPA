#!/usr/bin/env bash
#SBATCH --job-name=grid_goal_delta_jepa_train
#SBATCH --partition=rtxpro6k,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

VARIANTS=(
  FB_online_noema_nogoal
  FB_online_noema_goal
  FB_stopgrad_noema_nogoal
  FB_stopgrad_noema_goal
  FB_stopgrad_ema_nogoal
  FB_stopgrad_ema_goal
  SV_online_nogoal
  SV_online_goal
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

LATENT_REPRESENTATION="grid"
DYNAMICS_TARGET_MODE="online_no_stopgrad"
USE_EMA="false"
GOAL_MSE_WEIGHT="0.0"
GOAL_CONDITIONING="context_current"

case "${SELECTED_VARIANT}" in
  FB_online_noema_nogoal) ;;
  FB_online_noema_goal) GOAL_MSE_WEIGHT="1.0" ;;
  FB_stopgrad_noema_nogoal) DYNAMICS_TARGET_MODE="target_stopgrad" ;;
  FB_stopgrad_noema_goal) DYNAMICS_TARGET_MODE="target_stopgrad"; GOAL_MSE_WEIGHT="1.0" ;;
  FB_stopgrad_ema_nogoal) DYNAMICS_TARGET_MODE="target_stopgrad"; USE_EMA="true" ;;
  FB_stopgrad_ema_goal) DYNAMICS_TARGET_MODE="target_stopgrad"; USE_EMA="true"; GOAL_MSE_WEIGHT="1.0" ;;
  SV_online_nogoal) LATENT_REPRESENTATION="single"; GOAL_CONDITIONING="context_current" ;;
  SV_online_goal) LATENT_REPRESENTATION="single"; GOAL_CONDITIONING="context_current"; GOAL_MSE_WEIGHT="1.0" ;;
  *)
    echo "Unknown variant ${SELECTED_VARIANT}" >&2
    exit 2
    ;;
esac

RUN_SUFFIX="${RUN_SUFFIX:-}"
RUN_NAME="grid_goal_delta_jepa_${SELECTED_VARIANT}${RUN_SUFFIX}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/grid_goal_delta_jepa/${RUN_NAME}"

USER_OVERRIDES=()
if [[ -n "${EXTRA_HYDRA_OVERRIDES:-}" ]]; then
  read -r -a USER_OVERRIDES <<< "${EXTRA_HYDRA_OVERRIDES}"
fi

python -m puzzle_jepa.train.grid_goal_sudoku \
  --config-name grid_goal_sudoku \
  "seed=${SEED:-6206}" \
  "ablation=M0_full" \
  "+experiment_variant=${SELECTED_VARIANT}${RUN_SUFFIX}" \
  "output_dir=${RUN_ROOT}" \
  "task.train_limit=${TRAIN_LIMIT:-null}" \
  "task.eval_limit=${EVAL_LIMIT:-null}" \
  "training.max_steps=${TRAIN_MAX_STEPS:-5000}" \
  "training.batch_size=${BATCH_SIZE:-8}" \
  "training.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS:-1}" \
  "training.eval_every_steps=${EVAL_EVERY_STEPS:-500}" \
  "training.save_every_steps=${SAVE_EVERY_STEPS:-5000}" \
  "training.learning_rate=${LEARNING_RATE:-1.0e-4}" \
  "training.warmup_steps=${WARMUP_STEPS:-500}" \
  "training.bf16=${BF16:-true}" \
  "model.dropout=0.0" \
  "model.latent_representation=${LATENT_REPRESENTATION}" \
  "model.max_history_steps=${MAX_HISTORY_STEPS:-128}" \
  "model.action_conditioning=affected_marker" \
  "model.predict_delta=false" \
  "model.dynamics_target_mode=${DYNAMICS_TARGET_MODE}" \
  "model.dynamics_weighting=uniform" \
  "model.regularizer=none" \
  "model.sigreg_weight=0.0" \
  "model.use_ema_target_encoder=${USE_EMA}" \
  "model.ema_decay=${EMA_DECAY:-0.995}" \
  "model.delta_action_weight=${DELTA_ACTION_WEIGHT:-10.0}" \
  "model.delta_action_horizons=[1,2,3,4,5]" \
  "model.delta_action_decoder_layers=3" \
  "model.delta_action_max_steps=5" \
  "model.goal_mse_weight=${GOAL_MSE_WEIGHT}" \
  "model.goal_nce_weight=0.0" \
  "model.goal_distance_field_weight=0.0" \
  "model.goal_target_mode=online_no_stopgrad" \
  "model.goal_conditioning=${GOAL_CONDITIONING}" \
  "model.goal_conditioning_detach_state=false" \
  "model.dense_future_weight=${DENSE_FUTURE_WEIGHT:-1.0}" \
  "model.dense_rollout_all_steps=true" \
  "model.dense_rollout_variable_starts=false" \
  "model.dense_rollout_refactor_mode=none" \
  "model.dense_rollout_weighting=smooth_count" \
  "model.dense_rollout_gamma=${DENSE_ROLLOUT_GAMMA:-0.8}" \
  "model.multi_step_horizons=[8]" \
  "model.hierarchy_levels=[]" \
  "model.hierarchy_loss_weight=0.0" \
  "model.hierarchy_dense_future_weight=0.0" \
  "model.progress_rank_target=none" \
  "model.progress_rank_weight=0.0" \
  "model.action_rank_mode=none" \
  "model.action_rank_weight=0.0" \
  "model.terminal_corrupt_weight=0.0" \
  "model.policy_prior_weight=0.0" \
  "model.policy_prior_planning_weight=0.0" \
  "model.temporal_straightening_weight=0.0" \
  "${USER_OVERRIDES[@]}"
