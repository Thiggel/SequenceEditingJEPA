#!/usr/bin/env bash
#SBATCH --job-name=puzzle_grid4y_mixed_eval
#SBATCH --partition=a40,rtxpro6k
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --array=0-19%6
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

RUN_SPECS=(
  "sudoku_jepa_5m_global_mlp_l1_mixed_rollout_k2 1"
  "sudoku_jepa_5m_global_mlp_l1_mixed_rollout_k4 1"
  "sudoku_jepa_5m_global_mlp_l2_mixed_hierarchy_span4 2"
  "sudoku_jepa_5m_global_mlp_l3_mixed_hierarchy_span4 3"
)

EVAL_SPECS=()
for RUN_INDEX in "${!RUN_SPECS[@]}"; do
  read -r RUN_NAME LEVELS <<< "${RUN_SPECS[$RUN_INDEX]}"
  for HORIZON in 16 32; do
    EVAL_SPECS+=("${RUN_INDEX} mpc_cem goal_energy ${HORIZON} 0")
    EVAL_SPECS+=("${RUN_INDEX} mpc_cem latent_goal ${HORIZON} 0")
  done
  if (( LEVELS > 1 )); then
    TOP_LEVEL=$(( LEVELS - 1 ))
    EVAL_SPECS+=("${RUN_INDEX} recursive goal_energy 0 ${TOP_LEVEL}")
    EVAL_SPECS+=("${RUN_INDEX} recursive latent_goal 0 ${TOP_LEVEL}")
  fi
done

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if (( INDEX < 0 || INDEX >= ${#EVAL_SPECS[@]} )); then
  echo "Unknown SLURM_ARRAY_TASK_ID=${INDEX}; valid range is 0-$(( ${#EVAL_SPECS[@]} - 1 ))" >&2
  exit 2
fi

read -r RUN_INDEX EVAL_KIND SCORE HORIZON LEVEL <<< "${EVAL_SPECS[$INDEX]}"
read -r RUN_NAME LEVELS <<< "${RUN_SPECS[$RUN_INDEX]}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/${RUN_NAME}"

COMMON_ARGS=(
  --run-root "$RUN_ROOT"
  --rank-examples 0
  --drift-examples 0
  --trace-examples 0
  --planning-examples 0
  --cem-examples 0
  --mcts-examples 0
  --max-unroll-steps 81
  --seed $((6300 + INDEX))
)

case "${EVAL_KIND}" in
  mpc_cem)
    OUT_DIR="${RUN_ROOT}/diagnostics_mpc_cem_${SCORE}_h${HORIZON}"
    python -m puzzle_jepa.eval.diagnostics \
      "${COMMON_ARGS[@]}" \
      --output-dir "$OUT_DIR" \
      --mpc-cem-examples 64 \
      --mpc-cem-score "$SCORE" \
      --mpc-cem-horizon "$HORIZON" \
      --mpc-cem-execute-steps 1 \
      --mpc-cem-population 128 \
      --mpc-cem-iterations 4 \
      --mpc-cem-elite-frac 0.2 \
      --mpc-cem-smoothing 0.7
    ;;
  recursive)
    if (( LEVEL <= 0 )); then
      echo "Recursive eval requires LEVEL > 0." >&2
      exit 2
    fi
    if (( LEVEL == 1 )); then
      MACRO_HORIZON=12
    else
      MACRO_HORIZON=5
    fi
    OUT_DIR="${RUN_ROOT}/diagnostics_recursive_l${LEVEL}_${SCORE}_cem"
    python -m puzzle_jepa.eval.diagnostics \
      "${COMMON_ARGS[@]}" \
      --output-dir "$OUT_DIR" \
      --recursive-subgoal-examples 24 \
      --recursive-hierarchy-level "$LEVEL" \
      --recursive-macro-horizon "$MACRO_HORIZON" \
      --recursive-high-population 96 \
      --recursive-low-population 96 \
      --recursive-iterations 3 \
      --recursive-elite-frac 0.2 \
      --recursive-smoothing 0.7 \
      --recursive-execute-steps 1 \
      --recursive-prior-samples 32 \
      --recursive-high-score "$SCORE" \
      --recursive-optimizer cem
    ;;
  *)
    echo "Unknown EVAL_KIND=${EVAL_KIND}" >&2
    exit 2
    ;;
esac
