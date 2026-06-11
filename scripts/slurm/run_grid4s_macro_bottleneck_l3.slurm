#!/usr/bin/env bash
#SBATCH --job-name=puzzle_grid4s_macro
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-9%4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

# HWM-style macro-action bottleneck training grid. Planner evaluation is kept in
# run_grid4t_macro_bottleneck_planner_eval.slurm so it can be rerun without
# retraining checkpoints.
SPECS=(
  "terminal_energy 4 0 871"
  "terminal_energy 8 0 872"
  "terminal_energy 16 0 873"
  "terminal_energy 32 0 874"
  "terminal_energy 4 64 876"
  "terminal_energy 8 128 877"
  "state_value 4 0 878"
  "state_value 8 0 879"
  "macro_action_advantage 4 0 880"
  "macro_action_advantage 8 0 881"
)

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
read -r METHOD MACRO_DIM VQ_SIZE SEED <<< "${SPECS[$INDEX]}"

SUFFIX="d${MACRO_DIM}"
if [[ "${VQ_SIZE}" != "0" ]]; then
  SUFFIX="${SUFFIX}_vq${VQ_SIZE}"
fi
RUN_NAME="sudoku_jepa_5m_hier_macro_bottleneck_l3_${METHOD}_${SUFFIX}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/${RUN_NAME}"

COMMON_OVERRIDES=(
  "seed=${SEED}"
  "output_dir=${RUN_ROOT}"
  "model.hierarchy_levels=3"
  "model.hierarchy_span=4"
  "+model.macro_action_dim=${MACRO_DIM}"
  "training.hierarchy_weight=1.0"
  "training.hierarchy_rollout_steps=16"
  "training.hierarchy_batch_size=256"
  "+training.goal_energy_aux_sampling=stratified_puzzle"
  "+training.goal_energy_aux_puzzles=16"
  "+training.goal_energy_aux_states_per_puzzle=1"
)

if [[ "${VQ_SIZE}" != "0" ]]; then
  COMMON_OVERRIDES+=(
    "+model.use_macro_action_vq=true"
    "+model.macro_action_codebook_size=${VQ_SIZE}"
    "+model.macro_action_vq_beta=0.25"
  )
fi

METHOD_OVERRIDES=()
case "${METHOD}" in
  terminal_energy)
    METHOD_OVERRIDES=(
      "training.goal_energy_weight=1.0"
    )
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
    ;;
  macro_action_advantage)
    METHOD_OVERRIDES=(
      "training.goal_energy_weight=0.0"
      "+model.use_macro_action_value_head=true"
      "+training.macro_action_advantage_weight=1.0"
      "+training.macro_action_advantage_level=2"
      "+training.macro_action_advantage_batch_size=256"
      "+training.macro_action_advantage_standardize=true"
    )
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
