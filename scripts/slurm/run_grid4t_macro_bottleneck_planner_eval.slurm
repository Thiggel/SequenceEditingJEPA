#!/usr/bin/env bash
#SBATCH --job-name=puzzle_grid4t_plan
#SBATCH --partition=a40,rtxpro6k
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --array=0-57%6
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

# Each run is a 3-level checkpoint. Evaluation covers:
# - flat symbolic MCTS at depth 32, both learned scorer (where meaningful) and
#   oracle latent-goal control;
# - recursive L1 hierarchy planning with CEM;
# - recursive full L2/3-level hierarchy planning with CEM;
# each recursive level is evaluated with learned top score and oracle
# latent-goal top score.
RUN_SPECS=(
  "terminal_energy 4 0 goal_energy goal_energy"
  "terminal_energy 8 0 goal_energy goal_energy"
  "terminal_energy 16 0 goal_energy goal_energy"
  "terminal_energy 32 0 goal_energy goal_energy"
  "terminal_energy 4 64 goal_energy goal_energy"
  "terminal_energy 8 128 goal_energy goal_energy"
  "state_value 4 0 goal_value goal_value"
  "state_value 8 0 goal_value goal_value"
  "macro_action_advantage 4 0 macro_action_advantage skip"
  "macro_action_advantage 8 0 macro_action_advantage skip"
)

EVAL_SPECS=()
for RUN_INDEX in "${!RUN_SPECS[@]}"; do
  read -r METHOD MACRO_DIM VQ_SIZE HIGH_SCORE MCTS_SCORE <<< "${RUN_SPECS[$RUN_INDEX]}"
  EVAL_SPECS+=("${RUN_INDEX} mcts latent_goal 0")
  if [[ "${MCTS_SCORE}" != "skip" ]]; then
    EVAL_SPECS+=("${RUN_INDEX} mcts ${MCTS_SCORE} 0")
  fi
  EVAL_SPECS+=("${RUN_INDEX} recursive latent_goal 1")
  EVAL_SPECS+=("${RUN_INDEX} recursive ${HIGH_SCORE} 1")
  EVAL_SPECS+=("${RUN_INDEX} recursive latent_goal 2")
  EVAL_SPECS+=("${RUN_INDEX} recursive ${HIGH_SCORE} 2")
done

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if (( INDEX < 0 || INDEX >= ${#EVAL_SPECS[@]} )); then
  echo "Unknown SLURM_ARRAY_TASK_ID=${INDEX}; valid range is 0-$(( ${#EVAL_SPECS[@]} - 1 ))" >&2
  exit 2
fi

read -r RUN_INDEX EVAL_KIND SCORE LEVEL <<< "${EVAL_SPECS[$INDEX]}"
read -r METHOD MACRO_DIM VQ_SIZE HIGH_SCORE MCTS_SCORE <<< "${RUN_SPECS[$RUN_INDEX]}"

SUFFIX="d${MACRO_DIM}"
if [[ "${VQ_SIZE}" != "0" ]]; then
  SUFFIX="${SUFFIX}_vq${VQ_SIZE}"
fi
RUN_NAME="sudoku_jepa_5m_hier_macro_bottleneck_l3_${METHOD}_${SUFFIX}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/${RUN_NAME}"

COMMON_ARGS=(
  --run-root "$RUN_ROOT"
  --rank-examples 0
  --drift-examples 0
  --trace-examples 0
  --planning-examples 0
  --cem-examples 0
  --max-unroll-steps 81
  --seed $((5200 + INDEX))
)

case "${EVAL_KIND}" in
  mcts)
    OUT_DIR="${RUN_ROOT}/diagnostics_mcts_${SCORE}_d32_s256"
    python -m puzzle_jepa.eval.diagnostics \
      "${COMMON_ARGS[@]}" \
      --output-dir "$OUT_DIR" \
      --mcts-examples 16 \
      --mcts-score "$SCORE" \
      --mcts-simulations 256 \
      --mcts-depth 32 \
      --mcts-exploration 1.0 \
      --mcts-expansion-actions 32 \
      --mcts-debug-examples 6 \
      --mcts-debug-actions 16
    ;;
  recursive)
    if [[ "${LEVEL}" == "1" ]]; then
      MACRO_HORIZON=12
    elif [[ "${LEVEL}" == "2" ]]; then
      MACRO_HORIZON=5
    else
      echo "Unsupported recursive LEVEL=${LEVEL}" >&2
      exit 2
    fi
    OUT_DIR="${RUN_ROOT}/diagnostics_recursive_l${LEVEL}_${SCORE}_cem"
    python -m puzzle_jepa.eval.diagnostics \
      "${COMMON_ARGS[@]}" \
      --output-dir "$OUT_DIR" \
      --recursive-subgoal-examples 16 \
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
      --recursive-optimizer cem \
      --recursive-gd-steps 24 \
      --recursive-gd-lr 0.03 \
      --recursive-reachability-weight 0.01
    ;;
  *)
    echo "Unknown EVAL_KIND=${EVAL_KIND}" >&2
    exit 2
    ;;
esac
