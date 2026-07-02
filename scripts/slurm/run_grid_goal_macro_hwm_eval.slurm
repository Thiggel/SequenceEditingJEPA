#!/usr/bin/env bash
#SBATCH --job-name=grid_goal_macro_hwm_eval
#SBATCH --partition=rtxpro6k,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

VARIANTS=(
  D4_H4_16
  D8_H4_16
  D16_H4_16
  D8_H4_16_32
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

EVAL_MODE="${EVAL_MODE:-baseline}"
RUN_NAME="grid_goal_macro_hwm_${SELECTED_VARIANT}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/grid_goal_macro_hwm/${RUN_NAME}"
CHECKPOINT="${RUN_ROOT}/checkpoint.pt"
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi

PLANNERS="hierarchical_cem"
OUTPUT_DIR="${RUN_ROOT}/planner_eval_${EVAL_MODE}"
HIGH_OPTIMIZER="cem"
HIGH_CODEBOOK="none"
case "${EVAL_MODE}" in
  baseline)
    PLANNERS="mpc_beam"
    ;;
  cem_none)
    HIGH_OPTIMIZER="cem"
    HIGH_CODEBOOK="none"
    ;;
  cem_codebook)
    HIGH_OPTIMIZER="cem"
    HIGH_CODEBOOK="init"
    ;;
  mppi_none)
    HIGH_OPTIMIZER="mppi"
    HIGH_CODEBOOK="none"
    ;;
  mppi_codebook)
    HIGH_OPTIMIZER="mppi"
    HIGH_CODEBOOK="init"
    ;;
  *)
    echo "Unknown EVAL_MODE=${EVAL_MODE}" >&2
    exit 2
    ;;
esac

DIAGNOSTIC_ARGS=()
if [[ "${SKIP_DIAGNOSTICS:-1}" == "1" ]]; then
  DIAGNOSTIC_ARGS=(--skip-diagnostics)
fi

python -m puzzle_jepa.eval.grid_goal_planner_matrix \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --examples "${EVAL_EXAMPLES:-8}" \
  --beam-widths "${BEAM_WIDTHS:-16}" \
  --beam-depths "${BEAM_DEPTHS:-4,16}" \
  --scores "${SCORES:-oracle_goal_raw_euclidean_distance}" \
  --transitions "${TRANSITIONS:-latent_rollout}" \
  --planners "${PLANNERS}" \
  --cem-samples "${CEM_SAMPLES:-32}" \
  --cem-iters "${CEM_ITERS:-2}" \
  --cem-elites "${CEM_ELITES:-8}" \
  --cem-momentum "${CEM_MOMENTUM:-0.7}" \
  --high-cem-samples "${HIGH_CEM_SAMPLES:-32}" \
  --high-cem-iters "${HIGH_CEM_ITERS:-2}" \
  --high-cem-elites "${HIGH_CEM_ELITES:-8}" \
  --high-cem-momentum "${HIGH_CEM_MOMENTUM:-0.7}" \
  --high-cem-std "${HIGH_CEM_STD:-1.0}" \
  --high-cem-optimizer "${HIGH_OPTIMIZER}" \
  --high-cem-temperature "${HIGH_CEM_TEMPERATURE:-1.0}" \
  --high-cem-codebook "${HIGH_CODEBOOK}" \
  --high-cem-codebook-size "${HIGH_CEM_CODEBOOK_SIZE:-64}" \
  --seed "${EVAL_SEED:-5301}" \
  "${DIAGNOSTIC_ARGS[@]}"
