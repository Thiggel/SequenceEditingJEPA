#!/usr/bin/env bash
#SBATCH --job-name=grid_goal_act_train
#SBATCH --partition=rtxpro6k,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-95%32
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

BASE_ABLATIONS=(R4_no_goal_nce R7_no_terminal_corrupt)
ACTION_VARIANTS=(
  A0_action_token
  A1_affected_marker
  A2_local_action_feature
  A3_action_cross_attention
  A4_adaln_action
  A5_action_token_delta
  A6_affected_marker_delta
  A7_local_action_feature_delta
)
STABILITY_VARIANTS=(S0_sigreg S3_ema_sigreg S4_ema_vicreg)
DYNAMICS_VARIANTS=(D0_uniform D1_affected)

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
TOTAL=$(( ${#BASE_ABLATIONS[@]} * ${#ACTION_VARIANTS[@]} * ${#STABILITY_VARIANTS[@]} * ${#DYNAMICS_VARIANTS[@]} ))
if (( INDEX < 0 || INDEX >= TOTAL )); then
  echo "Unknown SLURM_ARRAY_TASK_ID=${INDEX}; valid range is 0-$(( TOTAL - 1 ))" >&2
  exit 2
fi

dyn_index=$(( INDEX % ${#DYNAMICS_VARIANTS[@]} ))
tmp=$(( INDEX / ${#DYNAMICS_VARIANTS[@]} ))
stab_index=$(( tmp % ${#STABILITY_VARIANTS[@]} ))
tmp=$(( tmp / ${#STABILITY_VARIANTS[@]} ))
action_index=$(( tmp % ${#ACTION_VARIANTS[@]} ))
base_index=$(( tmp / ${#ACTION_VARIANTS[@]} ))

ABLATION="${BASE_ABLATIONS[$base_index]}"
ACTION_VARIANT="${ACTION_VARIANTS[$action_index]}"
STABILITY_VARIANT="${STABILITY_VARIANTS[$stab_index]}"
DYNAMICS_VARIANT="${DYNAMICS_VARIANTS[$dyn_index]}"

ACTION_CONDITIONING="action_token"
PREDICT_DELTA="false"
case "${ACTION_VARIANT}" in
  A0_action_token) ACTION_CONDITIONING="action_token" ;;
  A1_affected_marker) ACTION_CONDITIONING="affected_marker" ;;
  A2_local_action_feature) ACTION_CONDITIONING="local_action_feature" ;;
  A3_action_cross_attention) ACTION_CONDITIONING="action_cross_attention" ;;
  A4_adaln_action) ACTION_CONDITIONING="adaln_action" ;;
  A5_action_token_delta) ACTION_CONDITIONING="action_token"; PREDICT_DELTA="true" ;;
  A6_affected_marker_delta) ACTION_CONDITIONING="affected_marker"; PREDICT_DELTA="true" ;;
  A7_local_action_feature_delta) ACTION_CONDITIONING="local_action_feature"; PREDICT_DELTA="true" ;;
  *) echo "Unknown action variant ${ACTION_VARIANT}" >&2; exit 2 ;;
esac

REGULARIZER="sigreg"
USE_EMA="false"
case "${STABILITY_VARIANT}" in
  S0_sigreg) REGULARIZER="sigreg"; USE_EMA="false" ;;
  S3_ema_sigreg) REGULARIZER="sigreg"; USE_EMA="true" ;;
  S4_ema_vicreg) REGULARIZER="vicreg"; USE_EMA="true" ;;
  *) echo "Unknown stability variant ${STABILITY_VARIANT}" >&2; exit 2 ;;
esac

DYNAMICS_WEIGHTING="uniform"
if [[ "${DYNAMICS_VARIANT}" == "D1_affected" ]]; then
  DYNAMICS_WEIGHTING="affected"
fi

SEED=$(( 4101 + INDEX ))
RUN_NAME="grid_goal_action_${ABLATION}_${ACTION_VARIANT}_${STABILITY_VARIANT}_${DYNAMICS_VARIANT}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/grid_goal_action_suite/${RUN_NAME}"

python -m puzzle_jepa.train.grid_goal_sudoku \
  --config-name grid_goal_sudoku \
  "seed=${SEED}" \
  "ablation=${ABLATION}" \
  "output_dir=${RUN_ROOT}" \
  "training.max_steps=${TRAIN_MAX_STEPS:-60000}" \
  "training.batch_size=${BATCH_SIZE:-8}" \
  "training.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS:-1}" \
  "training.eval_every_steps=${EVAL_EVERY_STEPS:-1000}" \
  "training.save_every_steps=${SAVE_EVERY_STEPS:-5000}" \
  "training.learning_rate=${LEARNING_RATE:-1.0e-4}" \
  "model.action_conditioning=${ACTION_CONDITIONING}" \
  "model.predict_delta=${PREDICT_DELTA}" \
  "model.dynamics_weighting=${DYNAMICS_WEIGHTING}" \
  "model.affected_dynamics_weight=${AFFECTED_DYNAMICS_WEIGHT:-32.0}" \
  "model.regularizer=${REGULARIZER}" \
  "model.use_ema_target_encoder=${USE_EMA}" \
  "model.ema_decay=${EMA_DECAY:-0.995}"
