#!/usr/bin/env bash
#SBATCH --job-name=grid_goal_h1dbg_train
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-5%3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

VARIANTS=(
  K14816_LR1e4
  K14816_LR5e5
  K14816_LR2e5
  K16_LR1e4
  K16_LR5e5
  K16_LR2e5
)

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if (( INDEX < 0 || INDEX >= ${#VARIANTS[@]} )); then
  echo "Unknown SLURM_ARRAY_TASK_ID=${INDEX}; valid range is 0-$(( ${#VARIANTS[@]} - 1 ))" >&2
  exit 2
fi

VARIANT="${VARIANTS[$INDEX]}"
RUN_NAME="grid_goal_h1_debug_${VARIANT}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/grid_goal_h1_debug/${RUN_NAME}"
SEED=$(( 7201 + INDEX ))

case "${VARIANT}" in
  K14816_*)
    HORIZONS="[1,4,8,16]"
    ;;
  K16_*)
    HORIZONS="[1,16]"
    ;;
  *)
    echo "Unknown H1 debug variant ${VARIANT}" >&2
    exit 2
    ;;
esac

case "${VARIANT}" in
  *_LR1e4)
    LR="1.0e-4"
    ;;
  *_LR5e5)
    LR="5.0e-5"
    ;;
  *_LR2e5)
    LR="2.0e-5"
    ;;
  *)
    echo "Unknown LR in H1 debug variant ${VARIANT}" >&2
    exit 2
    ;;
esac

python -m puzzle_jepa.train.grid_goal_sudoku \
  --config-name grid_goal_sudoku \
  "seed=${SEED}" \
  "ablation=R4_no_goal_nce" \
  "output_dir=${RUN_ROOT}" \
  "task.train_limit=${TRAIN_LIMIT:-null}" \
  "task.eval_limit=${EVAL_LIMIT:-null}" \
  "training.max_steps=${TRAIN_MAX_STEPS:-45000}" \
  "training.batch_size=${BATCH_SIZE:-8}" \
  "training.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS:-1}" \
  "training.eval_every_steps=${EVAL_EVERY_STEPS:-1000}" \
  "training.save_every_steps=${SAVE_EVERY_STEPS:-5000}" \
  "training.learning_rate=${LEARNING_RATE:-${LR}}" \
  "model.action_conditioning=affected_marker" \
  "model.predict_delta=true" \
  "model.dynamics_weighting=uniform" \
  "model.regularizer=vicreg" \
  "model.use_ema_target_encoder=true" \
  "model.ema_decay=${EMA_DECAY:-0.995}" \
  "model.goal_nce_weight=0.0" \
  "model.dense_future_weight=${DENSE_FUTURE_WEIGHT:-1.0}" \
  "model.hierarchy_levels=[4,16]" \
  "model.hierarchy_loss_weight=${HIERARCHY_LOSS_WEIGHT:-1.0}" \
  "model.hierarchy_dense_future_weight=0.0" \
  "model.multi_step_horizons=${HORIZONS}" \
  "model.goal_conditioning=context" \
  "model.progress_rank_target=predicted" \
  "model.action_rank_mode=pairwise" \
  "model.action_rank_target=predicted" \
  "model.policy_prior_weight=0.0" \
  "model.policy_prior_planning_weight=0.0" \
  "model.temporal_straightening_weight=${TEMPORAL_STRAIGHTENING_WEIGHT:-0.1}"
