#!/usr/bin/env bash
#SBATCH --job-name=controlled_hwm
#SBATCH --partition=a40,rtxpro6k,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
unset VIRTUAL_ENV
export PATH="/usr/local/bin:/usr/bin:/bin"
unset XDG_CACHE_HOME HF_HOME HF_DATASETS_CACHE HUGGINGFACE_HUB_CACHE
unset TORCH_HOME TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR UV_CACHE_DIR
unset TMPDIR TMP TEMP
export PUZZLE_JEPA_WORK_ROOT="${PUZZLE_JEPA_WORK_ROOT:-/home/vault/$(id -gn)/${USER}/sequence-editing}"
export PUZZLE_JEPA_SCRATCH_ROOT="${PUZZLE_JEPA_WORK_ROOT}/scratch"
source scripts/env.sh

RUN_NAME="${RUN_NAME:?RUN_NAME is required}"
SEED="${SEED:?SEED is required}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
MODEL_CONFIG="${MODEL_CONFIG:-mlp_causal_hwm}"
OBJECTIVE_CONFIG="${OBJECTIVE_CONFIG:-ema_vicreg}"
LEVEL_SPANS="${LEVEL_SPANS:-[1]}"
PREDICTOR_ARCHITECTURE="${PREDICTOR_ARCHITECTURE:-transformer}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-1}"
ROLLOUT_ALL_LEVELS="${ROLLOUT_ALL_LEVELS:-true}"
ROLLOUT_LAMBDA="${ROLLOUT_LAMBDA:-1.0}"
TRAIN_FROM_LEVEL="${TRAIN_FROM_LEVEL:-0}"

INIT_ARGS=()
if [[ -n "${INIT_CHECKPOINT:-}" ]]; then
  INIT_ARGS+=("training.init_checkpoint=${INIT_CHECKPOINT}")
fi
MODEL_ARGS=()
if [[ -n "${HIDDEN_DIM:-}" ]]; then
  MODEL_ARGS+=("model.hidden_dim=${HIDDEN_DIM}")
fi
if [[ -n "${LDAD_HORIZON:-}" ]]; then
  MODEL_ARGS+=("model.ldad_horizon=${LDAD_HORIZON}")
fi
OBJECTIVE_ARGS=()
if [[ -n "${LDAD_WEIGHT:-}" ]]; then
  OBJECTIVE_ARGS+=("objective.ldad_weight=${LDAD_WEIGHT}")
fi
if [[ -n "${VICREG_VARIANCE_WEIGHT:-}" ]]; then
  OBJECTIVE_ARGS+=("objective.vicreg_variance_weight=${VICREG_VARIANCE_WEIGHT}")
fi
if [[ -n "${VICREG_COVARIANCE_WEIGHT:-}" ]]; then
  OBJECTIVE_ARGS+=("objective.vicreg_covariance_weight=${VICREG_COVARIANCE_WEIGHT}")
fi
if [[ -n "${VICREG_WEIGHT:-}" ]]; then
  OBJECTIVE_ARGS+=("objective.vicreg_weight=${VICREG_WEIGHT}")
fi
if [[ -n "${SIGREG_WEIGHT:-}" ]]; then
  OBJECTIVE_ARGS+=("objective.sigreg_weight=${SIGREG_WEIGHT}")
fi
if [[ -n "${SIGREG_NUM_SLICES:-}" ]]; then
  OBJECTIVE_ARGS+=("objective.sigreg_num_slices=${SIGREG_NUM_SLICES}")
fi
if [[ -n "${TARGET_MODE:-}" ]]; then
  OBJECTIVE_ARGS+=("objective.target_mode=${TARGET_MODE}")
fi
if [[ -n "${STOP_GRADIENT_TARGETS:-}" ]]; then
  OBJECTIVE_ARGS+=("objective.stop_gradient_targets=${STOP_GRADIENT_TARGETS}")
fi
if [[ -n "${PREDICTION_WEIGHT:-}" ]]; then
  OBJECTIVE_ARGS+=("objective.prediction_weight=${PREDICTION_WEIGHT}")
fi
if [[ -n "${OBJECTIVE_NAME:-}" ]]; then
  OBJECTIVE_ARGS+=("objective.name=${OBJECTIVE_NAME}")
fi

python -m puzzle_jepa.train.controlled_objects \
  "model=${MODEL_CONFIG}" \
  "objective=${OBJECTIVE_CONFIG}" \
  "seed=${SEED}" \
  "run_name=${RUN_NAME}" \
  "model.level_spans=${LEVEL_SPANS}" \
  "model.predictor_architecture=${PREDICTOR_ARCHITECTURE}" \
  "model.rollout_steps=${ROLLOUT_STEPS}" \
  "model.rollout_all_levels=${ROLLOUT_ALL_LEVELS}" \
  "model.rollout_lambda=${ROLLOUT_LAMBDA}" \
  "training.train_from_level=${TRAIN_FROM_LEVEL}" \
  "training.max_steps=${MAX_STEPS:-20000}" \
  "training.eval_every_steps=${EVAL_EVERY_STEPS:-5000}" \
  "training.batch_size=${BATCH_SIZE:-64}" \
  "training.train_trajectories=${TRAIN_TRAJECTORIES:-512}" \
  "data.object_count=${OBJECT_COUNT:-4}" \
  "data.trajectory_length=${TRAJECTORY_LENGTH:-512}" \
  "eval.trajectories=${EVAL_TRAJECTORIES:-128}" \
  "eval.planning_episodes=${PLANNING_EPISODES:-8}" \
  "eval.planning_candidates=${PLANNING_CANDIDATES:-32}" \
  "hydra.run.dir=${OUTPUT_ROOT}/${RUN_NAME}/hydra" \
  "output_dir=${OUTPUT_ROOT}/${RUN_NAME}" \
  "${MODEL_ARGS[@]}" \
  "${OBJECTIVE_ARGS[@]}" \
  "${INIT_ARGS[@]}"
