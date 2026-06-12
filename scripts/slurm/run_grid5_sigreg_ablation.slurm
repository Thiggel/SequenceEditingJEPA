#!/usr/bin/env bash
#SBATCH --job-name=puzzle_grid5_sigreg
#SBATCH --partition=a40,a100,rtxpro6k
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --array=0-23%24
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

SPECS=()
for ENCODER in mlp cls_transformer; do
  for PREDICTOR in mlp ar_transformer; do
    for DELTA in state delta; do
      for LATENT in 32 64 128; do
        SPECS+=("${ENCODER} ${PREDICTOR} ${DELTA} ${LATENT}")
      done
    done
  done
done

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if (( INDEX < 0 || INDEX >= ${#SPECS[@]} )); then
  echo "Unknown SLURM_ARRAY_TASK_ID=${INDEX}; valid range is 0-$(( ${#SPECS[@]} - 1 ))" >&2
  exit 2
fi

read -r ENCODER PREDICTOR DELTA LATENT <<< "${SPECS[$INDEX]}"
PREDICT_DELTA=false
if [[ "${DELTA}" == "delta" ]]; then
  PREDICT_DELTA=true
fi
HIDDEN=$(( LATENT * 4 ))
if (( HIDDEN < 64 )); then
  HIDDEN=64
fi
SEED=$(( 1100 + INDEX ))
RUN_NAME="grid5_sigreg_${ENCODER}_${PREDICTOR}_${DELTA}_z${LATENT}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/${RUN_NAME}"

python -m puzzle_jepa.train.grid5 \
  --config-name grid5_sudoku_sigreg \
  "seed=${SEED}" \
  "output_dir=${RUN_ROOT}" \
  "model.encoder_type=${ENCODER}" \
  "model.predictor_type=${PREDICTOR}" \
  "model.predict_delta=${PREDICT_DELTA}" \
  "model.latent_size=${LATENT}" \
  "model.encoder_hidden_size=${HIDDEN}" \
  "model.predictor_hidden_size=${HIDDEN}" \
  "model.transformer_layers=2" \
  "model.predictor_layers=2" \
  "model.action_size=16" \
  "model.sigreg_weight=1.0" \
  "training.goal_energy_weight=1.0"

python -m puzzle_jepa.eval.grid5_diagnostics \
  --run-root "$RUN_ROOT" \
  --output-dir "$RUN_ROOT/diagnostics" \
  --seed $(( 5100 + INDEX )) \
  --latent-examples 64 \
  --trajectory-examples 32 \
  --rank-examples 32 \
  --detail-examples 4 \
  --plan-examples 16 \
  --plan-beam-size 4 \
  --plan-branch-size 8 \
  --max-plan-steps 96
