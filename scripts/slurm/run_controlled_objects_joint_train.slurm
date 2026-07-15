#!/usr/bin/env bash
#SBATCH --job-name=co_joint_train
#SBATCH --partition=a40,rtxpro6k,a100,a100mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
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

TASK_MANIFEST="${TASK_MANIFEST:?TASK_MANIFEST is required}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
ROW="$(awk -F '\t' -v task_id="${TASK_ID}" 'NR > 1 && $1 == task_id {print; exit}' "${TASK_MANIFEST}")"
if [[ -z "${ROW}" ]]; then
  printf 'No task %s in %s.\n' "${TASK_ID}" "${TASK_MANIFEST}" >&2
  exit 1
fi
IFS=$'\t' read -r _ objective seed <<<"${ROW}"

target_mode=shared
stop_gradient=false
prediction_weight=1.0
vicreg_weight=0.0
sigreg_weight=0.0
ldad_weight=0.0
case "${objective}" in
  online) ;;
  ema) target_mode=ema; stop_gradient=true ;;
  sigreg) prediction_weight=0.995; sigreg_weight=0.005 ;;
  ema_sigreg) target_mode=ema; stop_gradient=true; prediction_weight=0.995; sigreg_weight=0.005 ;;
  vicreg) vicreg_weight=1.0 ;;
  ema_vicreg) target_mode=ema; stop_gradient=true; vicreg_weight=1.0 ;;
  ldad) ldad_weight=0.1 ;;
  ema_ldad) target_mode=ema; stop_gradient=true; ldad_weight=0.1 ;;
  vicreg_ldad) vicreg_weight=1.0; ldad_weight=0.1 ;;
  ema_vicreg_ldad) target_mode=ema; stop_gradient=true; vicreg_weight=1.0; ldad_weight=0.1 ;;
  sigreg_ldad) prediction_weight=0.995; sigreg_weight=0.005; ldad_weight=0.1 ;;
  ema_sigreg_ldad) target_mode=ema; stop_gradient=true; prediction_weight=0.995; sigreg_weight=0.005; ldad_weight=0.1 ;;
  *) printf 'Unknown objective: %s\n' "${objective}" >&2; exit 1 ;;
esac

run_name="${objective}_s${seed}_joint_h110100"
RUN_NAME="${run_name}" \
SEED="${seed}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
MODEL_CONFIG=mlp_hwm_1_10_100 \
OBJECTIVE_CONFIG=joint_objective \
OBJECTIVE_NAME="${objective}" \
LEVEL_SPANS='[1,10,100]' \
PREDICTOR_ARCHITECTURE=transformer \
ROLLOUT_STEPS=4 \
ROLLOUT_ALL_LEVELS=true \
ROLLOUT_LAMBDA=1.0 \
TRAIN_FROM_LEVEL=0 \
TARGET_MODE="${target_mode}" \
STOP_GRADIENT_TARGETS="${stop_gradient}" \
PREDICTION_WEIGHT="${prediction_weight}" \
VICREG_WEIGHT="${vicreg_weight}" \
VICREG_VARIANCE_WEIGHT=0.05 \
VICREG_COVARIANCE_WEIGHT=17.866 \
SIGREG_WEIGHT="${sigreg_weight}" \
SIGREG_NUM_SLICES="${SIGREG_NUM_SLICES:-1024}" \
LDAD_WEIGHT="${ldad_weight}" \
LDAD_HORIZON=1 \
HIDDEN_DIM=256 \
OBJECT_COUNT=2 \
TRAJECTORY_LENGTH=512 \
MAX_STEPS="${MAX_STEPS:-20000}" \
BATCH_SIZE="${BATCH_SIZE:-64}" \
TRAIN_TRAJECTORIES="${TRAIN_TRAJECTORIES:-512}" \
EVAL_TRAJECTORIES="${EVAL_TRAJECTORIES:-128}" \
PLANNING_EPISODES=0 \
PLANNING_CANDIDATES=512 \
bash scripts/slurm/run_controlled_objects_train.slurm
