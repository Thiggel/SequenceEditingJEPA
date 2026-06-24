#!/usr/bin/env bash
#SBATCH --job-name=grid_goal_fu_train
#SBATCH --partition=rtxpro6k,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-5%6
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

VARIANTS=(
  F0_dense_k16
  F1_dense_k32_detach8
  H0_hierarchy_l4_l16
  H1_hierarchy_dense_l4_l16
  S0_scale_d384_dense
  S1_deeper_d256_dense
)

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if (( INDEX < 0 || INDEX >= ${#VARIANTS[@]} )); then
  echo "Unknown SLURM_ARRAY_TASK_ID=${INDEX}; valid range is 0-$(( ${#VARIANTS[@]} - 1 ))" >&2
  exit 2
fi

VARIANT="${VARIANTS[$INDEX]}"
RUN_NAME="grid_goal_followup_${VARIANT}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/grid_goal_followups/${RUN_NAME}"
SEED=$(( 5201 + INDEX ))

COMMON_OVERRIDES=(
  --config-name grid_goal_sudoku
  "seed=${SEED}"
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
  "model.regularizer=vicreg"
  "model.use_ema_target_encoder=true"
  "model.ema_decay=${EMA_DECAY:-0.995}"
)

case "${VARIANT}" in
  F0_dense_k16)
    EXTRA_OVERRIDES=(
      "model.dense_future_weight=${DENSE_FUTURE_WEIGHT:-1.0}"
      "model.multi_step_horizons=[1,4,8,16]"
    )
    ;;
  F1_dense_k32_detach8)
    EXTRA_OVERRIDES=(
      "model.dense_future_weight=${DENSE_FUTURE_WEIGHT:-1.0}"
      "model.multi_step_horizons=[1,4,8,16,32]"
      "model.rollout_detach_interval=${ROLLOUT_DETACH_INTERVAL:-8}"
    )
    ;;
  H0_hierarchy_l4_l16)
    EXTRA_OVERRIDES=(
      "model.hierarchy_levels=[4,16]"
      "model.hierarchy_loss_weight=${HIERARCHY_LOSS_WEIGHT:-1.0}"
      "model.multi_step_horizons=[1,4,8,16]"
    )
    ;;
  H1_hierarchy_dense_l4_l16)
    EXTRA_OVERRIDES=(
      "model.dense_future_weight=${DENSE_FUTURE_WEIGHT:-1.0}"
      "model.hierarchy_levels=[4,16]"
      "model.hierarchy_loss_weight=${HIERARCHY_LOSS_WEIGHT:-1.0}"
      "model.multi_step_horizons=[1,4,8,16]"
    )
    ;;
  S0_scale_d384_dense)
    EXTRA_OVERRIDES=(
      "training.batch_size=${SCALE_BATCH_SIZE:-4}"
      "model.d_model=384"
      "model.distance_dim=192"
      "model.num_heads=8"
      "model.dense_future_weight=${DENSE_FUTURE_WEIGHT:-1.0}"
      "model.multi_step_horizons=[1,4,8,16]"
    )
    ;;
  S1_deeper_d256_dense)
    EXTRA_OVERRIDES=(
      "model.context_layers=6"
      "model.state_layers=8"
      "model.predictor_layers=6"
      "model.goal_layers=8"
      "model.dense_future_weight=${DENSE_FUTURE_WEIGHT:-1.0}"
      "model.multi_step_horizons=[1,4,8,16]"
    )
    ;;
  *)
    echo "Unknown follow-up variant ${VARIANT}" >&2
    exit 2
    ;;
esac

python -m puzzle_jepa.train.grid_goal_sudoku "${COMMON_OVERRIDES[@]}" "${EXTRA_OVERRIDES[@]}"
