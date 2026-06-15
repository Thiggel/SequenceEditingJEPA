#!/usr/bin/env bash
#SBATCH --job-name=lewm_sudoku_plan
#SBATCH --partition=a100,a40,rtxpro6k
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-24%25
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

LRS=(
  1e-6 2e-6 3e-6 4e-6 5e-6 6e-6 7e-6 8e-6 9e-6
  1e-5 2e-5 3e-5 4e-5 5e-5 6e-5 7e-5 8e-5 9e-5
  1e-4 2e-4 3e-4 4e-4 5e-4 6e-4 7e-4
)

PLANNER="${PLANNER:?Set PLANNER to greedy, beam, best_first, categorical_cem, local_search, mcts, or exact.}"
TRANSITIONS="${TRANSITIONS:-symbolic_reencode,latent_rollout}"
SCORES="${SCORES:-true_hamming_oracle,oracle_goal_distance,predicted_goal_distance}"
DEPTHS="${DEPTHS:-4,8,16,32,64}"
EXAMPLES="${EXAMPLES:-4}"
FAST_FLAG="${FAST_FLAG:---fast}"

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if (( INDEX < 0 || INDEX >= ${#LRS[@]} )); then
  echo "Unknown SLURM_ARRAY_TASK_ID=${INDEX}; valid range is 0-$(( ${#LRS[@]} - 1 ))" >&2
  exit 2
fi

LR="${LRS[$INDEX]}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/lewm_sudoku_lr_${LR}"
CHECKPOINT="${RUN_ROOT}/checkpoint.pt"
OUTPUT_DIR="${RUN_ROOT}/posthoc_planners/${PLANNER}"

if [[ ! -s "${CHECKPOINT}" ]]; then
  echo "Missing checkpoint ${CHECKPOINT}; cannot run planner eval for ${PLANNER}." >&2
  exit 3
fi

if [[ -s "${OUTPUT_DIR}/planner_matrix.jsonl" ]]; then
  echo "Planner eval already exists for ${PLANNER} ${RUN_ROOT}; skipping."
  exit 0
fi

python -m puzzle_jepa.eval.lewm_planner_matrix \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --examples "${EXAMPLES}" \
  ${FAST_FLAG} \
  --skip-diagnostics \
  --planners "${PLANNER}" \
  --transitions "${TRANSITIONS}" \
  --scores "${SCORES}" \
  --depths "${DEPTHS}"
