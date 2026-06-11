#!/usr/bin/env bash
#SBATCH --job-name=puzzle_grid4v_global_eval
#SBATCH --partition=a40,rtxpro6k
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --array=0-15%6
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

RUN_SPECS=(
  "sudoku_jepa_5m_global_mlp_l1 1"
  "sudoku_jepa_5m_global_mlp_l2_span4 2"
  "sudoku_jepa_5m_global_mlp_l3_span4 3"
)

EVAL_SPECS=()
for RUN_INDEX in "${!RUN_SPECS[@]}"; do
  read -r RUN_NAME LEVELS <<< "${RUN_SPECS[$RUN_INDEX]}"
  EVAL_SPECS+=("${RUN_INDEX} mpc_cem goal_energy 0")
  EVAL_SPECS+=("${RUN_INDEX} mpc_cem latent_goal 0")
  EVAL_SPECS+=("${RUN_INDEX} reset_beam goal_energy 0")
  EVAL_SPECS+=("${RUN_INDEX} reset_beam latent_goal 0")
  if (( LEVELS > 1 )); then
    TOP_LEVEL=$(( LEVELS - 1 ))
    EVAL_SPECS+=("${RUN_INDEX} recursive goal_energy ${TOP_LEVEL}")
    EVAL_SPECS+=("${RUN_INDEX} recursive latent_goal ${TOP_LEVEL}")
  fi
done

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if (( INDEX < 0 || INDEX >= ${#EVAL_SPECS[@]} )); then
  echo "Unknown SLURM_ARRAY_TASK_ID=${INDEX}; valid range is 0-$(( ${#EVAL_SPECS[@]} - 1 ))" >&2
  exit 2
fi

read -r RUN_INDEX EVAL_KIND SCORE LEVEL <<< "${EVAL_SPECS[$INDEX]}"
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
  --seed $((6100 + INDEX))
)

case "${EVAL_KIND}" in
  mpc_cem)
    OUT_DIR="${RUN_ROOT}/diagnostics_mpc_cem_${SCORE}_h16"
    python -m puzzle_jepa.eval.diagnostics \
      "${COMMON_ARGS[@]}" \
      --output-dir "$OUT_DIR" \
      --mpc-cem-examples 64 \
      --mpc-cem-score "$SCORE" \
      --mpc-cem-horizon 16 \
      --mpc-cem-execute-steps 1 \
      --mpc-cem-population 128 \
      --mpc-cem-iterations 4 \
      --mpc-cem-elite-frac 0.2 \
      --mpc-cem-smoothing 0.7
    ;;
  reset_beam)
    OUT_DIR="${RUN_ROOT}/diagnostics_reset_beam_${SCORE}"
    python -m puzzle_jepa.eval.diagnostics \
      "${COMMON_ARGS[@]}" \
      --output-dir "$OUT_DIR" \
      --planning-examples 128 \
      --planning-score "$SCORE" \
      --planning-beam-size 4 \
      --planning-branch-size 8 \
      --reset-cadences 8 16
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
