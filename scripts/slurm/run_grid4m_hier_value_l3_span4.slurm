#!/usr/bin/env bash
#SBATCH --job-name=puzzle_grid4m_l3
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

METHODS=(
  terminal_energy
  action_advantage
  state_value
  contrastive_margin
)
SEEDS=(861 862 863 864)

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
METHOD="${METHODS[$INDEX]}"
SEED="${SEEDS[$INDEX]}"
RUN_NAME="sudoku_jepa_5m_hier_value_l3_span4_${METHOD}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/${RUN_NAME}"
PLANNING_SCORE="goal_energy"
HIGH_SCORE="goal_energy"

COMMON_OVERRIDES=(
  "seed=${SEED}"
  "output_dir=${RUN_ROOT}"
  "model.hierarchy_levels=3"
  "model.hierarchy_span=4"
  "training.hierarchy_weight=1.0"
  "training.hierarchy_rollout_steps=16"
  "training.hierarchy_batch_size=256"
  "+training.goal_energy_aux_sampling=stratified_puzzle"
  "+training.goal_energy_aux_puzzles=16"
  "+training.goal_energy_aux_states_per_puzzle=1"
)

METHOD_OVERRIDES=()
case "${METHOD}" in
  terminal_energy)
    METHOD_OVERRIDES=(
      "training.goal_energy_weight=1.0"
    )
    PLANNING_SCORE="goal_energy"
    HIGH_SCORE="goal_energy"
    ;;
  action_advantage)
    METHOD_OVERRIDES=(
      "training.goal_energy_weight=0.0"
      "+model.use_action_value_head=true"
      "+training.action_advantage_weight=1.0"
      "+training.action_advantage_standardize=true"
    )
    PLANNING_SCORE="action_advantage"
    HIGH_SCORE="latent_goal"
    ;;
  state_value)
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
    HIGH_SCORE="goal_value"
    ;;
  contrastive_margin)
    METHOD_OVERRIDES=(
      "training.goal_energy_weight=0.0"
      "+training.goal_energy_contrastive_loss=margin"
      "+training.goal_energy_contrastive_weight=1.0"
      "+training.goal_energy_aux_batch_size=64"
      "+training.goal_energy_aux_regression_weight=0.0"
      "+training.goal_energy_negative_count=8"
      "+training.goal_energy_contrastive_margin=0.05"
    )
    PLANNING_SCORE="goal_energy"
    HIGH_SCORE="goal_energy"
    ;;
  *)
    echo "Unknown METHOD=${METHOD}" >&2
    exit 2
    ;;
esac

python -m puzzle_jepa.train.grid0 \
  --config-name grid4a_sudoku_jepa_5m_goal_energy_cem_l3 \
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

python -m puzzle_jepa.eval.diagnostics \
  --run-root "$RUN_ROOT" \
  --output-dir "$RUN_ROOT/diagnostics_subgoal_l2_latent_goal" \
  --rank-examples 0 \
  --drift-examples 0 \
  --trace-examples 0 \
  --planning-examples 0 \
  --cem-examples 0 \
  --subgoal-cem-examples 32 \
  --subgoal-hierarchy-level 2 \
  --subgoal-macro-horizon 5 \
  --subgoal-high-population 256 \
  --subgoal-low-population 256 \
  --subgoal-iterations 4 \
  --subgoal-elite-frac 0.2 \
  --subgoal-smoothing 0.7 \
  --subgoal-execute-steps 1 \
  --subgoal-prior-samples 64 \
  --subgoal-high-score latent_goal \
  --max-unroll-steps 81

if [[ "${HIGH_SCORE}" != "latent_goal" ]]; then
  python -m puzzle_jepa.eval.diagnostics \
    --run-root "$RUN_ROOT" \
    --output-dir "$RUN_ROOT/diagnostics_subgoal_l2_${HIGH_SCORE}" \
    --rank-examples 0 \
    --drift-examples 0 \
    --trace-examples 0 \
    --planning-examples 0 \
    --cem-examples 0 \
    --subgoal-cem-examples 32 \
    --subgoal-hierarchy-level 2 \
    --subgoal-macro-horizon 5 \
    --subgoal-high-population 256 \
    --subgoal-low-population 256 \
    --subgoal-iterations 4 \
    --subgoal-elite-frac 0.2 \
    --subgoal-smoothing 0.7 \
    --subgoal-execute-steps 1 \
    --subgoal-prior-samples 64 \
    --subgoal-high-score "$HIGH_SCORE" \
    --max-unroll-steps 81
fi
