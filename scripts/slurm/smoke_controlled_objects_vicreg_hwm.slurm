#!/usr/bin/env bash
#SBATCH --job-name=co_vr_smoke
#SBATCH --partition=a40,rtxpro6k,a100,a100mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
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

output_root="${PUZZLE_JEPA_WORK_ROOT}/smoke/controlled_valid_hwm_vicreg"
rm -rf "${output_root}"

run_stage() {
  local run_name="$1"
  local spans="$2"
  local train_from_level="$3"
  local init_checkpoint="$4"
  RUN_NAME="${run_name}" \
  SEED=7 \
  OUTPUT_ROOT="${output_root}" \
  MODEL_CONFIG=mlp_hwm_1_10_100 \
  OBJECTIVE_CONFIG=ema_vicreg_sweep \
  LEVEL_SPANS="${spans}" \
  PREDICTOR_ARCHITECTURE=transformer \
  ROLLOUT_STEPS=4 \
  ROLLOUT_ALL_LEVELS=true \
  ROLLOUT_LAMBDA=1.0 \
  TRAIN_FROM_LEVEL="${train_from_level}" \
  INIT_CHECKPOINT="${init_checkpoint}" \
  HIDDEN_DIM=16 \
  OBJECT_COUNT=2 \
  TRAJECTORY_LENGTH=512 \
  VICREG_VARIANCE_WEIGHT=1 \
  VICREG_COVARIANCE_WEIGHT=1 \
  LDAD_WEIGHT=0 \
  MAX_STEPS=1 \
  EVAL_EVERY_STEPS=1 \
  BATCH_SIZE=2 \
  TRAIN_TRAJECTORIES=2 \
  EVAL_TRAJECTORIES=2 \
  PLANNING_EPISODES=0 \
  PLANNING_CANDIDATES=8 \
  bash scripts/slurm/run_controlled_objects_train.slurm
}

run_stage smoke_h1 '[1]' 0 ''
run_stage smoke_h110 '[1,10]' 1 "${output_root}/smoke_h1/checkpoint.pt"
run_stage smoke_h110100 '[1,10,100]' 2 "${output_root}/smoke_h110/checkpoint.pt"

test -s "${output_root}/smoke_h110100/checkpoint.pt"
printf 'Three-stage valid HWM smoke passed: %s\n' "${output_root}"
