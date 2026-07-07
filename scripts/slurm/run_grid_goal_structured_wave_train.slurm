#!/usr/bin/env bash
#SBATCH --job-name=grid_goal_struct_train
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
  S0_cell_baseline
  S1_unit_slots
  S2_global_slot
  S3_progress_slot
  S4_full_slots
  DJ0_action_token_all
  DJ1_marker_all
  DJ2_local_all
  DJ3_cross_all
  DJ4_marker_cell_units
  DJ5_cross_cell_units
  DJ0_action_token_all_single
  DJ1_marker_all_single
  DJ2_local_all_single
  DJ3_cross_all_single
  DJ4_marker_cell_units_single
  DJ5_cross_cell_units_single
  SD0_projection_only
  SD1_progress_rank
  SD2_action_subspace
  SD3_progress_action
  PR0_state_pair
  PR1_legal_listwise
  PR2_counterfactual_successor
  PR3_progress_slot_rank
  PR4_successor_progress
  GW0_terminal_goal
  GW1_waypoint_only
  GW2_waypoint_goal
  GW3_goal_conditioned_waypoint
  GW4_multi_waypoint
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

STRUCTURED_SLOTS="none"
LATENT_REPRESENTATION="grid"
ACTION_CONDITIONING="affected_marker"
DELTA_WEIGHT="0.0"
DELTA_SOURCE="all_tokens"
DELTA_HORIZONS="[1]"
DELTA_MODE="ordered"
SD_PROGRESS_WEIGHT="0.0"
PROGRESS_RANK_TARGET="none"
PROGRESS_RANK_WEIGHT="0.0"
ACTION_RANK_MODE="none"
ACTION_RANK_TARGET="predicted"
ACTION_RANK_WEIGHT="0.0"
WAYPOINT_HORIZONS="[]"
WAYPOINT_WEIGHT="0.0"
WAYPOINT_FINAL_WEIGHT="0.0"
GOAL_MSE_WEIGHT="0.0"
GOAL_NCE_WEIGHT="0.0"
GOAL_CONDITIONING="initial_current"
GOAL_DISTANCE_FIELD_WEIGHT="0.0"
WAYPOINT_CONDITIONING="current"
WAYPOINT_PLANNING_WEIGHT="1.0"
GOAL_PLANNING_WEIGHT="0.1"

case "${SELECTED_VARIANT}" in
  S0_cell_baseline)
    ;;
  S1_unit_slots)
    STRUCTURED_SLOTS="unit" ;;
  S2_global_slot)
    STRUCTURED_SLOTS="global" ;;
  S3_progress_slot)
    STRUCTURED_SLOTS="progress"; SD_PROGRESS_WEIGHT="0.25" ;;
  S4_full_slots)
    STRUCTURED_SLOTS="full"; SD_PROGRESS_WEIGHT="0.25" ;;
  DJ0_action_token_all)
    ACTION_CONDITIONING="action_token"; DELTA_WEIGHT="3.0"; DELTA_SOURCE="all_tokens" ;;
  DJ1_marker_all)
    ACTION_CONDITIONING="affected_marker"; DELTA_WEIGHT="3.0"; DELTA_SOURCE="all_tokens" ;;
  DJ2_local_all)
    ACTION_CONDITIONING="local_action_feature"; DELTA_WEIGHT="3.0"; DELTA_SOURCE="all_tokens" ;;
  DJ3_cross_all)
    ACTION_CONDITIONING="action_cross_attention"; DELTA_WEIGHT="3.0"; DELTA_SOURCE="all_tokens" ;;
  DJ4_marker_cell_units)
    STRUCTURED_SLOTS="full"; ACTION_CONDITIONING="affected_marker"; DELTA_WEIGHT="3.0"; DELTA_SOURCE="changed_cell_units" ;;
  DJ5_cross_cell_units)
    STRUCTURED_SLOTS="full"; ACTION_CONDITIONING="action_cross_attention"; DELTA_WEIGHT="3.0"; DELTA_SOURCE="changed_cell_units" ;;
  DJ0_action_token_all_single)
    LATENT_REPRESENTATION="single"; ACTION_CONDITIONING="action_token"; DELTA_WEIGHT="3.0"; DELTA_SOURCE="all_tokens" ;;
  DJ1_marker_all_single)
    LATENT_REPRESENTATION="single"; ACTION_CONDITIONING="affected_marker"; DELTA_WEIGHT="3.0"; DELTA_SOURCE="all_tokens" ;;
  DJ2_local_all_single)
    LATENT_REPRESENTATION="single"; ACTION_CONDITIONING="local_action_feature"; DELTA_WEIGHT="3.0"; DELTA_SOURCE="all_tokens" ;;
  DJ3_cross_all_single)
    LATENT_REPRESENTATION="single"; ACTION_CONDITIONING="action_cross_attention"; DELTA_WEIGHT="3.0"; DELTA_SOURCE="all_tokens" ;;
  DJ4_marker_cell_units_single)
    LATENT_REPRESENTATION="single"; ACTION_CONDITIONING="affected_marker"; DELTA_WEIGHT="3.0"; DELTA_SOURCE="all_tokens" ;;
  DJ5_cross_cell_units_single)
    LATENT_REPRESENTATION="single"; ACTION_CONDITIONING="action_cross_attention"; DELTA_WEIGHT="3.0"; DELTA_SOURCE="all_tokens" ;;
  SD0_projection_only)
    STRUCTURED_SLOTS="full"; SD_PROGRESS_WEIGHT="0.0" ;;
  SD1_progress_rank)
    STRUCTURED_SLOTS="full"; SD_PROGRESS_WEIGHT="1.0" ;;
  SD2_action_subspace)
    STRUCTURED_SLOTS="full"; DELTA_WEIGHT="3.0"; DELTA_SOURCE="changed_cell_units" ;;
  SD3_progress_action)
    STRUCTURED_SLOTS="full"; SD_PROGRESS_WEIGHT="1.0"; DELTA_WEIGHT="3.0"; DELTA_SOURCE="changed_cell_units" ;;
  PR0_state_pair)
    STRUCTURED_SLOTS="full"; PROGRESS_RANK_TARGET="oracle"; PROGRESS_RANK_WEIGHT="1.0" ;;
  PR1_legal_listwise)
    STRUCTURED_SLOTS="full"; ACTION_RANK_MODE="listwise"; ACTION_RANK_TARGET="oracle"; ACTION_RANK_WEIGHT="1.0" ;;
  PR2_counterfactual_successor)
    STRUCTURED_SLOTS="full"; ACTION_RANK_MODE="successor_pairwise"; ACTION_RANK_TARGET="oracle"; ACTION_RANK_WEIGHT="1.0" ;;
  PR3_progress_slot_rank)
    STRUCTURED_SLOTS="full"; SD_PROGRESS_WEIGHT="1.0"; PROGRESS_RANK_TARGET="oracle"; PROGRESS_RANK_WEIGHT="1.0" ;;
  PR4_successor_progress)
    STRUCTURED_SLOTS="full"; SD_PROGRESS_WEIGHT="1.0"; ACTION_RANK_MODE="listwise"; ACTION_RANK_TARGET="oracle"; ACTION_RANK_WEIGHT="1.0" ;;
  GW0_terminal_goal)
    STRUCTURED_SLOTS="full"; GOAL_MSE_WEIGHT="1.0"; GOAL_NCE_WEIGHT="0.1"; GOAL_CONDITIONING="initial_current" ;;
  GW1_waypoint_only)
    STRUCTURED_SLOTS="full"; WAYPOINT_HORIZONS="[8]"; WAYPOINT_WEIGHT="1.0"; WAYPOINT_FINAL_WEIGHT="0.1" ;;
  GW2_waypoint_goal)
    STRUCTURED_SLOTS="full"; GOAL_MSE_WEIGHT="0.25"; WAYPOINT_HORIZONS="[8]"; WAYPOINT_WEIGHT="1.0"; WAYPOINT_FINAL_WEIGHT="0.1"; WAYPOINT_PLANNING_WEIGHT="1.0"; GOAL_PLANNING_WEIGHT="0.1" ;;
  GW3_goal_conditioned_waypoint)
    STRUCTURED_SLOTS="full"; GOAL_MSE_WEIGHT="0.25"; WAYPOINT_HORIZONS="[8]"; WAYPOINT_WEIGHT="1.0"; WAYPOINT_FINAL_WEIGHT="0.1"; GOAL_CONDITIONING="initial_current"; WAYPOINT_CONDITIONING="current_goal"; GOAL_DISTANCE_FIELD_WEIGHT="0.25" ;;
  GW4_multi_waypoint)
    STRUCTURED_SLOTS="full"; GOAL_MSE_WEIGHT="0.25"; WAYPOINT_HORIZONS="[4,8,16]"; WAYPOINT_WEIGHT="1.0"; WAYPOINT_FINAL_WEIGHT="0.1"; WAYPOINT_PLANNING_WEIGHT="1.0"; GOAL_PLANNING_WEIGHT="0.1" ;;
  *)
    echo "Unknown variant ${SELECTED_VARIANT}" >&2
    exit 2
    ;;
esac

RUN_SUFFIX="${RUN_SUFFIX:-}"
RUN_NAME="grid_goal_structured_wave_${SELECTED_VARIANT}${RUN_SUFFIX}"
OUTPUT_ROOT="${PUZZLE_JEPA_WORK_ROOT:-${WORK:?WORK must be set}/sequence-editing-structured-20260707}"
RUN_ROOT="${OUTPUT_ROOT}/runs/grid_goal_structured_wave/${RUN_NAME}"

USER_OVERRIDES=()
if [[ -n "${EXTRA_HYDRA_OVERRIDES:-}" ]]; then
  read -r -a USER_OVERRIDES <<< "${EXTRA_HYDRA_OVERRIDES}"
fi

python -m puzzle_jepa.train.grid_goal_sudoku \
  --config-name grid_goal_sudoku \
  "seed=${SEED:-9707}" \
  "ablation=R4_no_goal_nce" \
  "+experiment_variant=${SELECTED_VARIANT}${RUN_SUFFIX}" \
  "output_dir=${RUN_ROOT}" \
  "task.train_limit=${TRAIN_LIMIT:-null}" \
  "task.eval_limit=${EVAL_LIMIT:-null}" \
  "training.max_steps=${TRAIN_MAX_STEPS:-5000}" \
  "training.batch_size=${BATCH_SIZE:-4}" \
  "training.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS:-2}" \
  "training.eval_every_steps=${EVAL_EVERY_STEPS:-500}" \
  "training.save_every_steps=${SAVE_EVERY_STEPS:-5000}" \
  "training.learning_rate=${LEARNING_RATE:-1.0e-4}" \
  "training.warmup_steps=${WARMUP_STEPS:-500}" \
  "training.bf16=${BF16:-true}" \
  "training.oracle_probability=${ORACLE_PROBABILITY:-0.7}" \
  "training.allow_overwrite=true" \
  "training.editable_noise_probability=${EDITABLE_NOISE_PROBABILITY:-0.15}" \
  "training.random_max_steps=${RANDOM_MAX_STEPS:-96}" \
  "training.counterfactual_branches=${COUNTERFACTUAL_BRANCHES:-16}" \
  "training.counterfactual_depth=${COUNTERFACTUAL_DEPTH:-8}" \
  "training.counterfactual_max_pairs=${COUNTERFACTUAL_MAX_PAIRS:-128}" \
  "model.dropout=0.0" \
  "model.latent_representation=${LATENT_REPRESENTATION}" \
  "model.structured_slots=${STRUCTURED_SLOTS}" \
  "model.action_conditioning=${ACTION_CONDITIONING}" \
  "model.predict_delta=false" \
  "model.dynamics_target_mode=target_stopgrad" \
  "model.dynamics_weighting=affected_context" \
  "model.affected_dynamics_weight=${AFFECTED_DYNAMICS_WEIGHT:-8.0}" \
  "model.context_dynamics_weight=${CONTEXT_DYNAMICS_WEIGHT:-2.0}" \
  "model.regularizer=vicreg" \
  "model.sigreg_weight=0.1" \
  "model.use_ema_target_encoder=true" \
  "model.ema_decay=${EMA_DECAY:-0.995}" \
  "model.goal_conditioning=${GOAL_CONDITIONING}" \
  "model.goal_mse_weight=${GOAL_MSE_WEIGHT}" \
  "model.goal_nce_weight=${GOAL_NCE_WEIGHT}" \
  "model.goal_distance_field_weight=${GOAL_DISTANCE_FIELD_WEIGHT}" \
  "model.waypoint_horizons=${WAYPOINT_HORIZONS}" \
  "model.waypoint_loss_weight=${WAYPOINT_WEIGHT}" \
  "model.waypoint_final_weight=${WAYPOINT_FINAL_WEIGHT}" \
  "model.waypoint_conditioning=${WAYPOINT_CONDITIONING}" \
  "model.waypoint_planning_weight=${WAYPOINT_PLANNING_WEIGHT}" \
  "model.goal_planning_weight=${GOAL_PLANNING_WEIGHT}" \
  "model.dense_future_weight=${DENSE_FUTURE_WEIGHT:-1.0}" \
  "model.dense_rollout_all_steps=true" \
  "model.dense_rollout_variable_starts=false" \
  "model.dense_rollout_refactor_mode=none" \
  "model.dense_rollout_weighting=smooth_count" \
  "model.multi_step_horizons=[8]" \
  "model.rollout_detach_interval=0" \
  "model.counterfactual_dynamics_weight=${COUNTERFACTUAL_DYNAMICS_WEIGHT:-1.0}" \
  "model.hierarchy_levels=[]" \
  "model.hierarchy_loss_weight=0.0" \
  "model.hierarchy_dense_future_weight=0.0" \
  "model.progress_rank_target=${PROGRESS_RANK_TARGET}" \
  "model.progress_rank_weight=${PROGRESS_RANK_WEIGHT}" \
  "model.action_rank_mode=${ACTION_RANK_MODE}" \
  "model.action_rank_target=${ACTION_RANK_TARGET}" \
  "model.action_rank_weight=${ACTION_RANK_WEIGHT}" \
  "model.listwise_action_rank_max_actions=${LISTWISE_ACTION_RANK_MAX_ACTIONS:-128}" \
  "model.temporal_straightening_weight=0.0" \
  "model.terminal_corrupt_weight=0.0" \
  "model.delta_action_weight=${DELTA_WEIGHT}" \
  "model.delta_action_horizons=${DELTA_HORIZONS}" \
  "model.delta_action_mode=${DELTA_MODE}" \
  "model.delta_action_source=${DELTA_SOURCE}" \
  "model.delta_action_decoder_layers=3" \
  "model.delta_action_max_steps=8" \
  "model.sd_progress_weight=${SD_PROGRESS_WEIGHT}" \
  "model.sd_progress_dim=${SD_PROGRESS_DIM:-32}" \
  "${USER_OVERRIDES[@]}"
