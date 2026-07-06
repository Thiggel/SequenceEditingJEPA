#!/usr/bin/env bash
#SBATCH --job-name=grid_goal_single_wide_train
#SBATCH --partition=rtxpro6k,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
source scripts/env.sh
mkdir -p logs

VARIANTS=(
  W0_ema_vicreg_d1024
  W1_ema_ldad_set_d1024
  W2_ldad_vicreg_set_d1024
  W3_ldad_only_set_d1024
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

RUN_NAME="grid_goal_single_wide_${SELECTED_VARIANT}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/grid_goal_single_wide/${RUN_NAME}"

REGULARIZER="vicreg"
SIGREG_WEIGHT="0.1"
USE_EMA="true"
DYNAMICS_TARGET_MODE="target_stopgrad"
DELTA_WEIGHT="0.0"
DELTA_HORIZONS="[1]"
DELTA_MODE="set"

case "${SELECTED_VARIANT}" in
  W0_ema_vicreg_d1024)
    ;;
  W1_ema_ldad_set_d1024)
    REGULARIZER="none"; SIGREG_WEIGHT="0.0"; DELTA_WEIGHT="10.0"; DELTA_HORIZONS="[1,2,3,4,5]" ;;
  W2_ldad_vicreg_set_d1024)
    USE_EMA="false"; DYNAMICS_TARGET_MODE="online_no_stopgrad"; DELTA_WEIGHT="10.0"; DELTA_HORIZONS="[1,2,3,4,5]" ;;
  W3_ldad_only_set_d1024)
    REGULARIZER="none"; SIGREG_WEIGHT="0.0"; USE_EMA="false"; DYNAMICS_TARGET_MODE="online_no_stopgrad"; DELTA_WEIGHT="10.0"; DELTA_HORIZONS="[1,2,3,4,5]" ;;
  *)
    echo "Unknown single-wide variant ${SELECTED_VARIANT}" >&2
    exit 2 ;;
esac

python -m puzzle_jepa.train.grid_goal_sudoku \
  --config-name grid_goal_sudoku \
  "ablation=R4_no_goal_nce" \
  "+experiment_variant=single_wide_${SELECTED_VARIANT}" \
  "output_dir=${RUN_ROOT}" \
  "seed=${SEED:-7901}" \
  "task.train_limit=${TRAIN_LIMIT:-null}" \
  "task.eval_limit=${EVAL_LIMIT:-null}" \
  "training.max_steps=${TRAIN_MAX_STEPS:-5000}" \
  "training.batch_size=${BATCH_SIZE:-2}" \
  "training.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS:-4}" \
  "training.eval_every_steps=${EVAL_EVERY_STEPS:-500}" \
  "training.save_every_steps=${SAVE_EVERY_STEPS:-5000}" \
  "training.learning_rate=${LEARNING_RATE:-1.0e-4}" \
  "training.warmup_steps=${WARMUP_STEPS:-500}" \
  "training.oracle_probability=${ORACLE_PROBABILITY:-0.7}" \
  "training.allow_overwrite=true" \
  "training.editable_noise_probability=${EDIT_NOISE:-0.15}" \
  "training.random_max_steps=${RANDOM_MAX_STEPS:-96}" \
  "training.counterfactual_branches=${CF_BRANCHES:-16}" \
  "training.counterfactual_depth=${CF_DEPTH:-4}" \
  "training.counterfactual_max_pairs=${CF_MAX_PAIRS:-128}" \
  "model.latent_representation=single" \
  "model.d_model=${D_MODEL:-1024}" \
  "model.distance_dim=${DISTANCE_DIM:-256}" \
  "model.num_heads=${NUM_HEADS:-16}" \
  "model.dropout=0.0" \
  "model.action_conditioning=affected_marker" \
  "model.predict_delta=false" \
  "model.multi_step_horizons=[8]" \
  "model.dense_rollout_all_steps=true" \
  "model.dense_rollout_weighting=smooth_count" \
  "model.dense_future_weight=1.0" \
  "model.dynamics_weighting=affected_context" \
  "model.counterfactual_dynamics_weight=1.0" \
  "model.regularizer=${REGULARIZER}" \
  "model.sigreg_weight=${SIGREG_WEIGHT}" \
  "model.use_ema_target_encoder=${USE_EMA}" \
  "model.dynamics_target_mode=${DYNAMICS_TARGET_MODE}" \
  "model.delta_action_weight=${DELTA_WEIGHT}" \
  "model.delta_action_horizons=${DELTA_HORIZONS}" \
  "model.delta_action_mode=${DELTA_MODE}" \
  "model.delta_action_decoder_layers=3" \
  "model.delta_action_max_steps=5" \
  "model.goal_mse_weight=0.0" \
  "model.goal_nce_weight=0.0" \
  "model.goal_distance_field_weight=0.0" \
  "model.waypoint_horizons=[]" \
  "model.waypoint_loss_weight=0.0" \
  "model.waypoint_final_weight=0.0" \
  "model.hierarchy_levels=[]" \
  "model.hierarchy_loss_weight=0.0" \
  "model.hierarchy_dense_future_weight=0.0" \
  "model.metric_geometry_mode=none" \
  "model.metric_geometry_weight=0.0" \
  "model.progress_rank_weight=0.0" \
  "model.action_rank_weight=0.0" \
  "model.temporal_straightening_weight=0.0" \
  "model.terminal_corrupt_weight=0.0"
