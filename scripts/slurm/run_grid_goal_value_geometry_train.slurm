#!/usr/bin/env bash
#SBATCH --job-name=grid_goal_value_train
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
  V0_base
  V1_hindsight_metric
  V2_iql_euclidean
  V3_iql_quasimetric
  V4_terminal_value
  V5_success_vector
  V6_success_vector_iql
  V7_success_vector_q
  V8_bad_state_iql_quasi
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

GOAL_MSE_WEIGHT="0.0"
GOAL_CONDITIONING="initial_current"
METRIC_GOAL_MSE_WEIGHT="0.0"
METRIC_GEOMETRY_MODE="none"
METRIC_GEOMETRY_WEIGHT="0.0"
METRIC_DISTANCE_TYPE="euclidean"
METRIC_ASYMMETRIC="false"
BAD_STATE_WEIGHT="0.0"
BAD_MARGIN_WEIGHT="0.0"
BAD_STATE_PLANNING_WEIGHT="0.0"
POLICY_PRIOR_WEIGHT="0.0"
POLICY_PRIOR_PLANNING_WEIGHT="0.0"

case "${SELECTED_VARIANT}" in
  V0_base)
    GOAL_MSE_WEIGHT="1.0"
    GOAL_CONDITIONING="context"
    ;;
  V1_hindsight_metric)
    METRIC_GOAL_MSE_WEIGHT="1.0"
    METRIC_GEOMETRY_MODE="hindsight"
    METRIC_GEOMETRY_WEIGHT="1.0"
    BAD_STATE_WEIGHT="0.1"
    BAD_MARGIN_WEIGHT="0.1"
    BAD_STATE_PLANNING_WEIGHT="0.1"
    ;;
  V2_iql_euclidean)
    METRIC_GOAL_MSE_WEIGHT="1.0"
    METRIC_GEOMETRY_MODE="iql"
    METRIC_GEOMETRY_WEIGHT="1.0"
    BAD_STATE_WEIGHT="0.1"
    BAD_MARGIN_WEIGHT="0.1"
    BAD_STATE_PLANNING_WEIGHT="0.1"
    ;;
  V3_iql_quasimetric)
    METRIC_GOAL_MSE_WEIGHT="1.0"
    METRIC_GEOMETRY_MODE="iql"
    METRIC_GEOMETRY_WEIGHT="1.0"
    METRIC_DISTANCE_TYPE="quasimetric"
    METRIC_ASYMMETRIC="true"
    BAD_STATE_WEIGHT="0.1"
    BAD_MARGIN_WEIGHT="0.1"
    BAD_STATE_PLANNING_WEIGHT="0.1"
    ;;
  V4_terminal_value)
    METRIC_GEOMETRY_MODE="terminal_value"
    METRIC_GEOMETRY_WEIGHT="1.0"
    BAD_STATE_WEIGHT="0.1"
    BAD_STATE_PLANNING_WEIGHT="0.1"
    ;;
  V5_success_vector)
    METRIC_GEOMETRY_MODE="success"
    METRIC_GEOMETRY_WEIGHT="1.0"
    BAD_STATE_WEIGHT="0.1"
    BAD_STATE_PLANNING_WEIGHT="0.1"
    ;;
  V6_success_vector_iql)
    METRIC_GEOMETRY_MODE="success_iql"
    METRIC_GEOMETRY_WEIGHT="1.0"
    BAD_STATE_WEIGHT="0.1"
    BAD_STATE_PLANNING_WEIGHT="0.1"
    ;;
  V7_success_vector_q)
    METRIC_GEOMETRY_MODE="success_iql"
    METRIC_GEOMETRY_WEIGHT="1.0"
    BAD_STATE_WEIGHT="0.1"
    BAD_STATE_PLANNING_WEIGHT="0.1"
    POLICY_PRIOR_WEIGHT="0.1"
    POLICY_PRIOR_PLANNING_WEIGHT="0.1"
    ;;
  V8_bad_state_iql_quasi)
    METRIC_GOAL_MSE_WEIGHT="1.0"
    METRIC_GEOMETRY_MODE="iql"
    METRIC_GEOMETRY_WEIGHT="1.0"
    METRIC_DISTANCE_TYPE="quasimetric"
    METRIC_ASYMMETRIC="true"
    BAD_STATE_WEIGHT="0.5"
    BAD_MARGIN_WEIGHT="0.5"
    BAD_STATE_PLANNING_WEIGHT="0.5"
    ;;
  *)
    echo "Unknown variant ${SELECTED_VARIANT}" >&2
    exit 2
    ;;
esac

RUN_SUFFIX="${RUN_SUFFIX:-}"
RUN_NAME="grid_goal_value_geometry_${SELECTED_VARIANT}${RUN_SUFFIX}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/grid_goal_value_geometry/${RUN_NAME}"

USER_OVERRIDES=()
if [[ -n "${EXTRA_HYDRA_OVERRIDES:-}" ]]; then
  read -r -a USER_OVERRIDES <<< "${EXTRA_HYDRA_OVERRIDES}"
fi

python -m puzzle_jepa.train.grid_goal_sudoku \
  --config-name grid_goal_sudoku \
  "seed=${SEED:-7401}" \
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
  "model.latent_representation=grid" \
  "model.action_conditioning=affected_marker" \
  "model.predict_delta=false" \
  "model.dynamics_target_mode=target_stopgrad" \
  "model.dynamics_weighting=uniform" \
  "model.affected_dynamics_weight=${AFFECTED_DYNAMICS_WEIGHT:-8.0}" \
  "model.context_dynamics_weight=${CONTEXT_DYNAMICS_WEIGHT:-2.0}" \
  "model.regularizer=none" \
  "model.sigreg_weight=0.0" \
  "model.use_ema_target_encoder=true" \
  "model.ema_decay=${EMA_DECAY:-0.995}" \
  "model.goal_mse_weight=${GOAL_MSE_WEIGHT}" \
  "model.goal_nce_weight=0.0" \
  "model.goal_distance_field_weight=0.0" \
  "model.goal_target_mode=target_stopgrad" \
  "model.goal_conditioning=${GOAL_CONDITIONING}" \
  "model.goal_conditioning_detach_state=false" \
  "model.metric_goal_mse_weight=${METRIC_GOAL_MSE_WEIGHT}" \
  "model.metric_geometry_mode=${METRIC_GEOMETRY_MODE}" \
  "model.metric_geometry_weight=${METRIC_GEOMETRY_WEIGHT}" \
  "model.metric_distance_type=${METRIC_DISTANCE_TYPE}" \
  "model.metric_asymmetric_projection=${METRIC_ASYMMETRIC}" \
  "model.metric_iql_gamma=${METRIC_IQL_GAMMA:-0.98}" \
  "model.metric_iql_expectile=${METRIC_IQL_EXPECTILE:-0.8}" \
  "model.metric_bad_margin_weight=${BAD_MARGIN_WEIGHT}" \
  "model.metric_bad_margin=${METRIC_BAD_MARGIN:-1.0}" \
  "model.metric_hindsight_max_horizon=${METRIC_HINDSIGHT_MAX_HORIZON:-8}" \
  "model.metric_contrastive_temperature=${METRIC_CONTRASTIVE_TEMPERATURE:-0.1}" \
  "model.bad_state_weight=${BAD_STATE_WEIGHT}" \
  "model.bad_state_planning_weight=${BAD_STATE_PLANNING_WEIGHT}" \
  "model.dense_future_weight=1.0" \
  "model.dense_rollout_all_steps=true" \
  "model.dense_rollout_variable_starts=false" \
  "model.dense_rollout_refactor_mode=none" \
  "model.dense_rollout_weighting=smooth_count" \
  "model.dense_rollout_gamma=${DENSE_ROLLOUT_GAMMA:-0.8}" \
  "model.multi_step_horizons=[8]" \
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
  "model.policy_prior_weight=${POLICY_PRIOR_WEIGHT}" \
  "model.policy_prior_mode=pairwise" \
  "model.policy_prior_planning_weight=${POLICY_PRIOR_PLANNING_WEIGHT}" \
  "model.temporal_straightening_weight=0.0" \
  "${USER_OVERRIDES[@]}"
