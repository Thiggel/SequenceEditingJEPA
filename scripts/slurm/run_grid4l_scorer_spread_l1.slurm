#!/usr/bin/env bash
#SBATCH --job-name=puzzle_grid4l_spread
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-6
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

METHODS=(
  scaled_energy
  action_advantage
  local_zscore
  local_margin
  task_value
  latent_progress
  muzero_value_mcts
)
SEEDS=(841 842 843 844 845 846 847)

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
METHOD="${METHODS[$INDEX]}"
SEED="${SEEDS[$INDEX]}"
RUN_NAME="sudoku_jepa_5m_scorer_spread_l1_${METHOD}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/${RUN_NAME}"
PLANNING_SCORE="goal_energy"

COMMON_OVERRIDES=(
  "seed=${SEED}"
  "output_dir=${RUN_ROOT}"
  "model.hierarchy_levels=1"
  "training.hierarchy_weight=0.0"
  "+training.goal_energy_aux_sampling=stratified_puzzle"
  "+training.goal_energy_aux_puzzles=16"
  "+training.goal_energy_aux_states_per_puzzle=1"
  "+training.goal_ranking_batch_size=16"
  "+training.goal_ranking_candidates=128"
)

METHOD_OVERRIDES=()
case "${METHOD}" in
  scaled_energy)
    METHOD_OVERRIDES=(
      "training.goal_energy_weight=1.0"
      "+training.goal_energy_target_scale=100.0"
    )
    PLANNING_SCORE="goal_energy"
    ;;
  action_advantage)
    METHOD_OVERRIDES=(
      "training.goal_energy_weight=0.0"
      "+model.use_action_value_head=true"
      "+training.action_advantage_weight=1.0"
      "+training.action_advantage_standardize=true"
    )
    PLANNING_SCORE="action_advantage"
    ;;
  local_zscore)
    METHOD_OVERRIDES=(
      "training.goal_energy_weight=0.0"
      "+training.goal_local_regression_weight=1.0"
      "+training.goal_local_regression_label=latent_goal_distance"
    )
    PLANNING_SCORE="goal_energy"
    ;;
  local_margin)
    METHOD_OVERRIDES=(
      "training.goal_energy_weight=0.0"
      "+training.goal_local_margin_weight=1.0"
      "+training.goal_local_margin=0.5"
    )
    PLANNING_SCORE="goal_energy"
    ;;
  task_value)
    METHOD_OVERRIDES=(
      "training.goal_energy_weight=0.0"
      "+training.goal_terminal_correctness_weight=1.0"
      "+training.goal_terminal_target_mode=discounted_reachability"
      "+training.goal_terminal_discount=0.99"
      "+training.goal_terminal_regression_weight=0.0"
      "+training.goal_terminal_batch_size=256"
      "+training.goal_terminal_positive_fraction=0.5"
    )
    PLANNING_SCORE="goal_value"
    ;;
  latent_progress)
    METHOD_OVERRIDES=(
      "training.goal_energy_weight=1.0"
      "+training.latent_progress_weight=1.0"
      "+training.latent_progress_batch_size=64"
      "+training.latent_progress_target_scale=0.25"
    )
    PLANNING_SCORE="goal_energy"
    ;;
  muzero_value_mcts)
    METHOD_OVERRIDES=(
      "training.goal_energy_weight=0.0"
      "+training.goal_terminal_correctness_weight=1.0"
      "+training.goal_terminal_target_mode=discounted_reachability"
      "+training.goal_terminal_discount=0.99"
      "+training.goal_terminal_regression_weight=0.0"
      "+training.goal_terminal_batch_size=256"
      "+training.goal_terminal_positive_fraction=0.5"
    )
    PLANNING_SCORE="goal_value"
    ;;
  *)
    echo "Unknown METHOD=${METHOD}" >&2
    exit 2
    ;;
esac

python -m puzzle_jepa.train.grid0 \
  --config-name grid4a_sudoku_jepa_5m_goal_energy_cem_l1 \
  "${COMMON_OVERRIDES[@]}" \
  "${METHOD_OVERRIDES[@]}"

python -m puzzle_jepa.eval.diagnostics \
  --run-root "$RUN_ROOT" \
  --output-dir "$RUN_ROOT/diagnostics_reset_${PLANNING_SCORE}" \
  --rank-examples 0 \
  --drift-examples 0 \
  --trace-examples 0 \
  --planning-examples 128 \
  --planning-score "$PLANNING_SCORE" \
  --planning-beam-size 4 \
  --planning-branch-size 8 \
  --reset-cadences 4 \
  --cem-examples 0 \
  --max-unroll-steps 81

python -m puzzle_jepa.eval.diagnostics \
  --run-root "$RUN_ROOT" \
  --output-dir "$RUN_ROOT/diagnostics_reset_oracle_calibration" \
  --rank-examples 0 \
  --drift-examples 0 \
  --trace-examples 0 \
  --planning-examples 128 \
  --planning-score latent_goal \
  --planning-beam-size 4 \
  --planning-branch-size 8 \
  --reset-cadences 4 \
  --cem-examples 0 \
  --max-unroll-steps 81

if [[ "${METHOD}" == "muzero_value_mcts" ]]; then
  python -m puzzle_jepa.eval.value_mcts \
    --run-root "$RUN_ROOT" \
    --output-dir "$RUN_ROOT/diagnostics_value_mcts" \
    --num-examples 64 \
    --seed "$((SEED + 1000))" \
    --max-steps 81 \
    --simulations 64 \
    --max-depth 4 \
    --max-actions 64 \
    --c-puct 1.0
fi
