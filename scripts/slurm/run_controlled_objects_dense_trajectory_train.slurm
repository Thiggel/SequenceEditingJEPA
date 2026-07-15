#!/usr/bin/env bash
#SBATCH --job-name=co_dense_traj
#SBATCH --partition=a40,rtxpro6k,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

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

ROW="$(awk -F '\t' -v task_id="${SLURM_ARRAY_TASK_ID:?}" 'NR > 1 && $1 == task_id {print; exit}' "${TASK_MANIFEST:?}")"
IFS=$'\t' read -r _ objective setting trajectory_length batch_size max_steps warmup_steps seed <<<"${ROW}"
max_steps="${MAX_STEPS_OVERRIDE:-${max_steps}}"
warmup_steps="${WARMUP_STEPS_OVERRIDE:-${warmup_steps}}"
target_mode=shared
stop_gradient=false
if [[ "${objective}" == ema_vicreg ]]; then
  target_mode=ema
  stop_gradient=true
fi
run_name="${objective}_${setting}_s${seed}_dense_h110100"

RUN_NAME="${run_name}" SEED="${seed}" OUTPUT_ROOT="${OUTPUT_ROOT}" \
MODEL_CONFIG=mlp_hwm_1_10_100 OBJECTIVE_CONFIG=joint_objective \
OBJECTIVE_NAME="dense_${objective}" LEVEL_SPANS='[1,10,100]' \
PREDICTOR_ARCHITECTURE=transformer PREDICTOR_MAX_CONTEXT=512 \
ROLLOUT_STEPS=4 ROLLOUT_STEPS_BY_LEVEL='[10,10,4]' \
ROLLOUT_ALL_LEVELS=true ROLLOUT_LAMBDA=1.0 \
DENSE_TRAJECTORY_TRAINING=true CROSS_LEVEL_CONSISTENCY_WEIGHT=1.0 \
TRAIN_FROM_LEVEL=0 TARGET_MODE="${target_mode}" \
STOP_GRADIENT_TARGETS="${stop_gradient}" PREDICTION_WEIGHT=1.0 \
VICREG_WEIGHT=1.0 VICREG_VARIANCE_WEIGHT=0.05 \
VICREG_COVARIANCE_WEIGHT=17.866 SIGREG_WEIGHT=0 LDAD_WEIGHT=0 \
HIDDEN_DIM=256 OBJECT_COUNT=2 TRAJECTORY_LENGTH="${trajectory_length}" \
MAX_STEPS="${max_steps}" EVAL_EVERY_STEPS="${max_steps}" \
WARMUP_STEPS="${warmup_steps}" BATCH_SIZE="${batch_size}" \
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}" \
TRAIN_TRAJECTORIES="${TRAIN_TRAJECTORIES:-512}" \
EVAL_TRAJECTORIES="${EVAL_TRAJECTORIES:-128}" PLANNING_EPISODES=0 \
bash scripts/slurm/run_controlled_objects_train.slurm
