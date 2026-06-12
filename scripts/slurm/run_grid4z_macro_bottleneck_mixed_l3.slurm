#!/usr/bin/env bash
#SBATCH --job-name=puzzle_grid4z_mixed
#SBATCH --partition=a40,rtxpro6k
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

# Focused rerun of the best Grid 4S oracle-MCTS variant, but with coherent
# mixed correct/wrong rollouts for both rollout loss and hierarchy loss.
RUN_NAME="sudoku_jepa_5m_hier_macro_bottleneck_l3_terminal_energy_d4_vq64_mixed_rollouts"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/${RUN_NAME}"

python -m puzzle_jepa.train.grid0 \
  --config-name grid4a_sudoku_jepa_5m_goal_energy_cem_l3 \
  "seed=921" \
  "output_dir=${RUN_ROOT}" \
  "model.hierarchy_levels=3" \
  "model.hierarchy_span=4" \
  "+model.macro_action_dim=4" \
  "+model.use_macro_action_vq=true" \
  "+model.macro_action_codebook_size=64" \
  "+model.macro_action_vq_beta=0.25" \
  "training.goal_energy_weight=1.0" \
  "training.hierarchy_weight=1.0" \
  "training.hierarchy_rollout_steps=16" \
  "training.hierarchy_batch_size=256" \
  "+training.rollout_oracle_probability=0.5" \
  "+training.hierarchy_oracle_probability=0.5" \
  "+training.goal_energy_aux_sampling=stratified_puzzle" \
  "+training.goal_energy_aux_puzzles=16" \
  "+training.goal_energy_aux_states_per_puzzle=1"

COMMON_ARGS=(
  --run-root "$RUN_ROOT"
  --rank-examples 0
  --drift-examples 0
  --trace-examples 0
  --planning-examples 0
  --cem-examples 0
  --max-unroll-steps 81
  --seed 9210
)

python -m puzzle_jepa.eval.diagnostics \
  "${COMMON_ARGS[@]}" \
  --output-dir "$RUN_ROOT/diagnostics_mcts_latent_goal_d32_s256" \
  --mcts-examples 16 \
  --mcts-score latent_goal \
  --mcts-simulations 256 \
  --mcts-depth 32 \
  --mcts-exploration 1.0 \
  --mcts-expansion-actions 32 \
  --mcts-debug-examples 6 \
  --mcts-debug-actions 16

python -m puzzle_jepa.eval.diagnostics \
  "${COMMON_ARGS[@]}" \
  --output-dir "$RUN_ROOT/diagnostics_mcts_goal_energy_d32_s256" \
  --mcts-examples 16 \
  --mcts-score goal_energy \
  --mcts-simulations 256 \
  --mcts-depth 32 \
  --mcts-exploration 1.0 \
  --mcts-expansion-actions 32 \
  --mcts-debug-examples 6 \
  --mcts-debug-actions 16

python -m puzzle_jepa.eval.diagnostics \
  "${COMMON_ARGS[@]}" \
  --output-dir "$RUN_ROOT/diagnostics_recursive_l2_latent_goal_cem" \
  --recursive-subgoal-examples 16 \
  --recursive-hierarchy-level 2 \
  --recursive-macro-horizon 5 \
  --recursive-high-population 96 \
  --recursive-low-population 96 \
  --recursive-iterations 3 \
  --recursive-elite-frac 0.2 \
  --recursive-smoothing 0.7 \
  --recursive-execute-steps 1 \
  --recursive-prior-samples 32 \
  --recursive-high-score latent_goal \
  --recursive-optimizer cem

python -m puzzle_jepa.eval.diagnostics \
  "${COMMON_ARGS[@]}" \
  --output-dir "$RUN_ROOT/diagnostics_recursive_l2_goal_energy_cem" \
  --recursive-subgoal-examples 16 \
  --recursive-hierarchy-level 2 \
  --recursive-macro-horizon 5 \
  --recursive-high-population 96 \
  --recursive-low-population 96 \
  --recursive-iterations 3 \
  --recursive-elite-frac 0.2 \
  --recursive-smoothing 0.7 \
  --recursive-execute-steps 1 \
  --recursive-prior-samples 32 \
  --recursive-high-score goal_energy \
  --recursive-optimizer cem
