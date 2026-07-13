#!/usr/bin/env bash
#SBATCH --job-name=grid_goal_weekend_train
#SBATCH --partition=rtxpro6k,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
source scripts/env.sh
mkdir -p logs

VARIANTS=(
  S0_anchor_olddata
  S1_counterfactual_fill
  S2_counterfactual_edit
  S3_counterfactual_edit_adaln
  S4_counterfactual_edit_oldlocal
  E0_base_cf_edit
  E1_hierarchy_l4_l8_l16
  E2_waypoint_h8
  E3_waypoint_h16
  E4_waypoint_h4_h8_h16
  E5_waypoint_h16_hierarchy
  D0_online_h1_single
  D1_online_ordered_h12345_single
  D2_online_set_h12345_single
  D3_hybrid_ema_h1_single
  D4_hybrid_ema_set_h12345_single
  D5_ema_vicreg_no_ldad_single
  V1_asym_hindsight
  V2_iql_quasi
  V3_waypoint_asym_hindsight
  I0_integrated_waypoint_asym
  I1_integrated_waypoint_iql
  I2_integrated_best_delta_if_gate_passes_single
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-${VARIANT_INDEX:-0}}"
VARIANT="${VARIANT:-${VARIANTS[$TASK_ID]}}"
RUN_NAME="grid_goal_weekend_${VARIANT}${RUN_SUFFIX:-}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/grid_goal_weekend/${RUN_NAME}"

LATENT="grid"
BASE_VARIANT="${VARIANT}"
if [[ "${VARIANT}" == *_single ]]; then
  LATENT="single"
  BASE_VARIANT="${VARIANT%_single}"
fi

ALLOW_OVERWRITE="true"
EDIT_NOISE="0.15"
CF_BRANCHES="16"
CF_DEPTH="4"
CF_MAX_PAIRS="128"
COUNTERFACTUAL_WEIGHT="1.0"
ACTION_CONDITIONING="affected_marker"
REGULARIZER="vicreg"
SIGREG_WEIGHT="0.1"
USE_EMA="true"
DYNAMICS_TARGET_MODE="target_stopgrad"
DELTA_WEIGHT="0.0"
DELTA_HORIZONS="[1]"
DELTA_MODE="ordered"
GOAL_MSE_WEIGHT="0.0"
GOAL_TARGET_MODE="target_stopgrad"
GOAL_CONDITIONING="context_current"
WAYPOINT_HORIZONS="[]"
WAYPOINT_WEIGHT="0.0"
WAYPOINT_FINAL_WEIGHT="0.0"
HIERARCHY_LEVELS="[]"
HIERARCHY_WEIGHT="0.0"
HIERARCHY_DENSE_WEIGHT="0.0"
METRIC_MODE="none"
METRIC_WEIGHT="0.0"
METRIC_ASYM="false"
METRIC_DISTANCE="euclidean"

case "${BASE_VARIANT}" in
  S0_anchor_olddata)
    ALLOW_OVERWRITE="false"; EDIT_NOISE="0.0"; CF_BRANCHES="0"; CF_MAX_PAIRS="0"; COUNTERFACTUAL_WEIGHT="0.0" ;;
  S1_counterfactual_fill)
    ALLOW_OVERWRITE="false"; EDIT_NOISE="0.0" ;;
  S2_counterfactual_edit|E0_base_cf_edit) ;;
  S3_counterfactual_edit_adaln)
    ACTION_CONDITIONING="affected_marker_adaln" ;;
  S4_counterfactual_edit_oldlocal)
    ACTION_CONDITIONING="old_local_concat" ;;
  E1_hierarchy_l4_l8_l16)
    HIERARCHY_LEVELS="[4,8,16]"; HIERARCHY_WEIGHT="1.0"; HIERARCHY_DENSE_WEIGHT="0.25"; CF_DEPTH="16"; CF_MAX_PAIRS="32" ;;
  E2_waypoint_h8)
    WAYPOINT_HORIZONS="[8]"; WAYPOINT_WEIGHT="1.0"; WAYPOINT_FINAL_WEIGHT="0.1" ;;
  E3_waypoint_h16)
    WAYPOINT_HORIZONS="[16]"; WAYPOINT_WEIGHT="1.0"; WAYPOINT_FINAL_WEIGHT="0.1" ;;
  E4_waypoint_h4_h8_h16)
    WAYPOINT_HORIZONS="[4,8,16]"; WAYPOINT_WEIGHT="1.0"; WAYPOINT_FINAL_WEIGHT="0.1" ;;
  E5_waypoint_h16_hierarchy)
    HIERARCHY_LEVELS="[4,8,16]"; HIERARCHY_WEIGHT="1.0"; HIERARCHY_DENSE_WEIGHT="0.25"; WAYPOINT_HORIZONS="[16]"; WAYPOINT_WEIGHT="1.0"; WAYPOINT_FINAL_WEIGHT="0.1"; CF_DEPTH="16"; CF_MAX_PAIRS="32" ;;
  V1_asym_hindsight)
    METRIC_MODE="hindsight"; METRIC_WEIGHT="1.0"; METRIC_ASYM="true" ;;
  V2_iql_quasi)
    METRIC_MODE="iql"; METRIC_WEIGHT="1.0"; METRIC_ASYM="true"; METRIC_DISTANCE="quasimetric" ;;
  V3_waypoint_asym_hindsight|I0_integrated_waypoint_asym)
    WAYPOINT_HORIZONS="[16]"; WAYPOINT_WEIGHT="1.0"; WAYPOINT_FINAL_WEIGHT="0.1"; METRIC_MODE="hindsight"; METRIC_WEIGHT="1.0"; METRIC_ASYM="true"; HIERARCHY_LEVELS="[4,8,16]"; HIERARCHY_WEIGHT="1.0"; HIERARCHY_DENSE_WEIGHT="0.25"; CF_DEPTH="16"; CF_MAX_PAIRS="32" ;;
  I1_integrated_waypoint_iql)
    WAYPOINT_HORIZONS="[16]"; WAYPOINT_WEIGHT="1.0"; WAYPOINT_FINAL_WEIGHT="0.1"; METRIC_MODE="iql"; METRIC_WEIGHT="1.0"; METRIC_ASYM="true"; METRIC_DISTANCE="quasimetric"; HIERARCHY_LEVELS="[4,8,16]"; HIERARCHY_WEIGHT="1.0"; HIERARCHY_DENSE_WEIGHT="0.25"; CF_DEPTH="16"; CF_MAX_PAIRS="32" ;;
  I2_integrated_best_delta_if_gate_passes)
    WAYPOINT_HORIZONS="[16]"; WAYPOINT_WEIGHT="1.0"; WAYPOINT_FINAL_WEIGHT="0.1"; HIERARCHY_LEVELS="[4,8,16]"; HIERARCHY_WEIGHT="1.0"; HIERARCHY_DENSE_WEIGHT="0.25"; DELTA_WEIGHT="10.0"; DELTA_HORIZONS="[1]"; REGULARIZER="none"; SIGREG_WEIGHT="0.0"; USE_EMA="false"; DYNAMICS_TARGET_MODE="online_no_stopgrad"; CF_DEPTH="16"; CF_MAX_PAIRS="32" ;;
  D0_online_h1)
    REGULARIZER="none"; SIGREG_WEIGHT="0.0"; USE_EMA="false"; DYNAMICS_TARGET_MODE="online_no_stopgrad"; DELTA_WEIGHT="10.0"; DELTA_HORIZONS="[1]" ;;
  D1_online_ordered_h12345)
    REGULARIZER="none"; SIGREG_WEIGHT="0.0"; USE_EMA="false"; DYNAMICS_TARGET_MODE="online_no_stopgrad"; DELTA_WEIGHT="10.0"; DELTA_HORIZONS="[1,2,3,4,5]" ;;
  D2_online_set_h12345)
    REGULARIZER="none"; SIGREG_WEIGHT="0.0"; USE_EMA="false"; DYNAMICS_TARGET_MODE="online_no_stopgrad"; DELTA_WEIGHT="10.0"; DELTA_HORIZONS="[1,2,3,4,5]"; DELTA_MODE="set" ;;
  D3_hybrid_ema_h1)
    REGULARIZER="none"; SIGREG_WEIGHT="0.0"; USE_EMA="true"; DYNAMICS_TARGET_MODE="target_stopgrad"; DELTA_WEIGHT="10.0"; DELTA_HORIZONS="[1]" ;;
  D4_hybrid_ema_set_h12345)
    REGULARIZER="none"; SIGREG_WEIGHT="0.0"; USE_EMA="true"; DYNAMICS_TARGET_MODE="target_stopgrad"; DELTA_WEIGHT="10.0"; DELTA_HORIZONS="[1,2,3,4,5]"; DELTA_MODE="set" ;;
  D5_ema_vicreg_no_ldad)
    DELTA_WEIGHT="0.0"; DELTA_HORIZONS="[1]" ;;
  *)
    echo "Unknown weekend variant ${VARIANT}" >&2
    exit 2 ;;
esac

python -m puzzle_jepa.train.grid_goal_sudoku \
  --config-name grid_goal_sudoku \
  "ablation=R4_no_goal_nce" \
  "+experiment_variant=${VARIANT}" \
  "output_dir=${RUN_ROOT}" \
  "seed=${SEED:-7601}" \
  "task.train_limit=${TRAIN_LIMIT:-null}" \
  "task.eval_limit=${EVAL_LIMIT:-null}" \
  "training.max_steps=${TRAIN_MAX_STEPS:-5000}" \
  "training.batch_size=${BATCH_SIZE:-4}" \
  "training.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS:-2}" \
  "training.eval_every_steps=${EVAL_EVERY_STEPS:-500}" \
  "training.save_every_steps=${SAVE_EVERY_STEPS:-5000}" \
  "training.learning_rate=${LEARNING_RATE:-1.0e-4}" \
  "training.warmup_steps=${WARMUP_STEPS:-500}" \
  "training.oracle_probability=${ORACLE_PROBABILITY:-0.7}" \
  "training.allow_overwrite=${ALLOW_OVERWRITE}" \
  "training.editable_noise_probability=${EDIT_NOISE}" \
  "training.random_max_steps=${RANDOM_MAX_STEPS:-96}" \
  "training.counterfactual_branches=${CF_BRANCHES}" \
  "training.counterfactual_depth=${CF_DEPTH}" \
  "training.counterfactual_max_pairs=${CF_MAX_PAIRS}" \
  "model.latent_representation=${LATENT}" \
  "model.dropout=0.0" \
  "model.action_conditioning=${ACTION_CONDITIONING}" \
  "model.predict_delta=false" \
  "model.multi_step_horizons=[8]" \
  "model.dense_rollout_all_steps=true" \
  "model.dense_rollout_weighting=smooth_count" \
  "model.dense_future_weight=1.0" \
  "model.dynamics_weighting=affected_context" \
  "model.counterfactual_dynamics_weight=${COUNTERFACTUAL_WEIGHT}" \
  "model.regularizer=${REGULARIZER}" \
  "model.sigreg_weight=${SIGREG_WEIGHT}" \
  "model.use_ema_target_encoder=${USE_EMA}" \
  "model.dynamics_target_mode=${DYNAMICS_TARGET_MODE}" \
  "model.delta_action_weight=${DELTA_WEIGHT}" \
  "model.delta_action_horizons=${DELTA_HORIZONS}" \
  "model.delta_action_mode=${DELTA_MODE}" \
  "model.delta_action_decoder_layers=3" \
  "model.delta_action_max_steps=5" \
  "model.goal_mse_weight=${GOAL_MSE_WEIGHT}" \
  "model.goal_nce_weight=0.0" \
  "model.goal_distance_field_weight=0.0" \
  "model.goal_target_mode=${GOAL_TARGET_MODE}" \
  "model.goal_conditioning=${GOAL_CONDITIONING}" \
  "model.waypoint_horizons=${WAYPOINT_HORIZONS}" \
  "model.waypoint_loss_weight=${WAYPOINT_WEIGHT}" \
  "model.waypoint_final_weight=${WAYPOINT_FINAL_WEIGHT}" \
  "model.hierarchy_levels=${HIERARCHY_LEVELS}" \
  "model.hierarchy_loss_weight=${HIERARCHY_WEIGHT}" \
  "model.hierarchy_dense_future_weight=${HIERARCHY_DENSE_WEIGHT}" \
  "model.shared_hierarchy_predictor=false" \
  "model.metric_geometry_mode=${METRIC_MODE}" \
  "model.metric_geometry_weight=${METRIC_WEIGHT}" \
  "model.metric_asymmetric_projection=${METRIC_ASYM}" \
  "model.metric_distance_type=${METRIC_DISTANCE}" \
  "model.progress_rank_weight=0.0" \
  "model.action_rank_weight=0.0" \
  "model.temporal_straightening_weight=0.0" \
  "model.terminal_corrupt_weight=0.0"
