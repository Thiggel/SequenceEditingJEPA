#!/bin/bash
#SBATCH --job-name=puzzle_grid5b_10m
#SBATCH --output=logs/puzzle_grid5b_10m_%A_%a.out
#SBATCH --error=logs/puzzle_grid5b_10m_%A_%a.err
#SBATCH --array=0-11%12
#SBATCH --partition=a40,a100,rtxpro6k
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00

set -euo pipefail

cd /home/hpc/c107fa/c107fa12/sequence-editing
source ./scripts/env.sh
mkdir -p logs

# name encoder predictor target stabilizer ema_momentum recursive_k
SPECS=(
  "canonical_sigreg_k4 cls_transformer ar_transformer state sigreg 0.0 4"
  "canonical_ema_sigreg_k4 cls_transformer ar_transformer state sigreg 0.99 4"
  "canonical_vicreg_k4 cls_transformer ar_transformer state vicreg 0.0 4"
  "canonical_ema_vicreg_k4 cls_transformer ar_transformer state vicreg 0.99 4"
  "canonical_sigreg_k1 cls_transformer ar_transformer state sigreg 0.0 1"
  "canonical_ema_sigreg_k1 cls_transformer ar_transformer state sigreg 0.99 1"
  "delta_sigreg_k4 cls_transformer ar_transformer delta sigreg 0.0 4"
  "mlp_pred_sigreg_k4 cls_transformer mlp state sigreg 0.0 4"
  "mlp_enc_sigreg_k4 mlp ar_transformer state sigreg 0.0 4"
  "oldbest_scaled_sigreg_k4 mlp mlp delta sigreg 0.0 4"
  "oldbest_scaled_ema_sigreg_k4 mlp mlp delta sigreg 0.99 4"
  "oldbest_scaled_sigreg_k1 mlp mlp delta sigreg 0.0 1"
)

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if (( INDEX < 0 || INDEX >= ${#SPECS[@]} )); then
  echo "Unknown SLURM_ARRAY_TASK_ID=${INDEX}; valid range is 0-$(( ${#SPECS[@]} - 1 ))" >&2
  exit 2
fi

read -r NAME ENCODER PREDICTOR TARGET STABILIZER EMA_MOMENTUM RECURSIVE_K <<<"${SPECS[$INDEX]}"
PREDICT_DELTA=false
if [[ "${TARGET}" == "delta" ]]; then
  PREDICT_DELTA=true
fi
RECURSIVE_WEIGHT=0.0
if [[ "${RECURSIVE_K}" -gt 1 ]]; then
  RECURSIVE_WEIGHT=1.0
fi

RUN_NAME="grid5b_10m_${NAME}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/${RUN_NAME}"
SEED=$(( 8100 + INDEX ))

python -m puzzle_jepa.train.grid5 --config-name grid5_sudoku_sigreg \
  seed="${SEED}" \
  output_dir="${RUN_ROOT}" \
  model.encoder_type="${ENCODER}" \
  model.predictor_type="${PREDICTOR}" \
  model.predict_delta="${PREDICT_DELTA}" \
  model.latent_size=512 \
  model.encoder_hidden_size=3072 \
  model.predictor_hidden_size=3072 \
  model.transformer_layers=1 \
  model.predictor_layers=1 \
  model.num_heads=8 \
  model.action_size=64 \
  model.max_rollout_steps=8 \
  model.stabilizer_type="${STABILIZER}" \
  model.sigreg_weight=1.0 \
  model.vicreg_variance_weight=1.0 \
  model.vicreg_covariance_weight=0.04 \
  model.target_encoder_momentum="${EMA_MOMENTUM}" \
  training.rollout_steps=8 \
  training.recursive_rollout_steps="${RECURSIVE_K}" \
  training.recursive_rollout_weight="${RECURSIVE_WEIGHT}" \
  training.goal_energy_weight=1.0 \
  training.max_steps=5000 \
  training.batch_size=512 \
  eval.batch_size=128

python -m puzzle_jepa.eval.grid5_diagnostics \
  --run-root "${RUN_ROOT}" \
  --output-dir "${RUN_ROOT}/diagnostics" \
  --seed "$(( 9100 + INDEX ))" \
  --latent-examples 64 \
  --trajectory-examples 32 \
  --rank-examples 32 \
  --detail-examples 4 \
  --plan-examples 16 \
  --plan-beam-size 4 \
  --plan-branch-size 8 \
  --max-plan-steps 96

python -m puzzle_jepa.eval.grid5_mpc_cem_diagnostics \
  --run-root "${RUN_ROOT}" \
  --output-dir "${RUN_ROOT}/diagnostics_mpc_cem" \
  --seed "$(( 10100 + INDEX ))" \
  --horizons 8 16 32 64 \
  --plan-examples 8 \
  --max-mpc-steps 64 \
  --candidates 300 \
  --elites 30 \
  --iterations 10 \
  --qualitative-examples 4 \
  --qualitative-candidates 6 \
  --qualitative-cem-candidates 128 \
  --qualitative-cem-iterations 5

python scripts/analysis/grid5_symbolic_planning_probe.py \
  --run-root "${RUN_NAME}=${RUN_ROOT}" \
  --output-dir "${RUN_ROOT}/diagnostics_symbolic_reencode" \
  --seed "$(( 11100 + INDEX ))" \
  --plan-examples 4 \
  --horizon 8 16 32 64 full \
  --score-mode latent_goal goal_energy true_hamming \
  --action-space fill_empty \
  --max-mpc-steps 64 \
  --candidates 128 \
  --elites 16 \
  --iterations 5 \
  --drift-horizon 1 2 4 8 16 32 \
  --drift-examples 64
