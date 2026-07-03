#!/usr/bin/env bash
#SBATCH --job-name=grid_goal_horizon_train
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
  K1_uniform
  K1_smooth_count
  K2_uniform
  K2_smooth_count
  K3_uniform
  K3_smooth_count
  K4_uniform
  K4_smooth_count
  K8_uniform
  K8_smooth_count
  K16_uniform
  K16_smooth_count
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

if [[ "${SELECTED_VARIANT}" =~ ^K([0-9]+)_(uniform|smooth_count)$ ]]; then
  HORIZON="${BASH_REMATCH[1]}"
  WEIGHTING="${BASH_REMATCH[2]}"
else
  echo "Unknown variant ${SELECTED_VARIANT}" >&2
  exit 2
fi

RUN_SUFFIX="${RUN_SUFFIX:-}"
RUN_NAME="grid_goal_horizon_ablation_${SELECTED_VARIANT}${RUN_SUFFIX}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/grid_goal_horizon_ablation/${RUN_NAME}"

USER_OVERRIDES=()
if [[ -n "${EXTRA_HYDRA_OVERRIDES:-}" ]]; then
  read -r -a USER_OVERRIDES <<< "${EXTRA_HYDRA_OVERRIDES}"
fi

python -m puzzle_jepa.train.grid_goal_sudoku \
  --config-name grid_goal_sudoku \
  "seed=${SEED:-5204}" \
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
  "model.action_conditioning=affected_marker" \
  "model.predict_delta=false" \
  "model.dynamics_weighting=uniform" \
  "model.affected_dynamics_weight=${AFFECTED_DYNAMICS_WEIGHT:-8.0}" \
  "model.context_dynamics_weight=${CONTEXT_DYNAMICS_WEIGHT:-2.0}" \
  "model.regularizer=none" \
  "model.use_ema_target_encoder=true" \
  "model.ema_decay=${EMA_DECAY:-0.995}" \
  "model.goal_mse_weight=1.0" \
  "model.goal_nce_weight=0.0" \
  "model.goal_distance_field_weight=0.0" \
  "model.goal_target_mode=target_stopgrad" \
  "model.goal_conditioning=context" \
  "model.goal_conditioning_detach_state=false" \
  "model.dense_future_weight=1.0" \
  "model.dense_rollout_all_steps=true" \
  "model.dense_rollout_variable_starts=false" \
  "model.dense_rollout_refactor_mode=none" \
  "model.dense_rollout_weighting=${WEIGHTING}" \
  "model.dense_rollout_gamma=${DENSE_ROLLOUT_GAMMA:-0.8}" \
  "model.multi_step_horizons=[${HORIZON}]" \
  "model.rollout_detach_interval=0" \
  "model.hierarchy_levels=[]" \
  "model.hierarchy_loss_weight=0.0" \
  "model.hierarchy_dense_future_weight=0.0" \
  "model.shared_hierarchy_predictor=false" \
  "model.progress_rank_target=none" \
  "model.progress_rank_weight=0.0" \
  "model.action_rank_mode=none" \
  "model.action_rank_target=predicted" \
  "model.action_rank_weight=0.0" \
  "model.terminal_corrupt_weight=0.0" \
  "model.policy_prior_weight=0.0" \
  "model.policy_prior_planning_weight=0.0" \
  "model.temporal_straightening_weight=0.0" \
  "${USER_OVERRIDES[@]}"
