#!/usr/bin/env bash
#SBATCH --job-name=puzzle_grid4n_macro
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

RUN_NAME="sudoku_jepa_5m_hier_value_l3_span4_macro_action_advantage"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/${RUN_NAME}"

python -m puzzle_jepa.train.grid0 \
  --config-name grid4a_sudoku_jepa_5m_goal_energy_cem_l3 \
  "seed=865" \
  "output_dir=${RUN_ROOT}" \
  "model.hierarchy_levels=3" \
  "model.hierarchy_span=4" \
  "+model.use_macro_action_value_head=true" \
  "training.hierarchy_weight=1.0" \
  "training.hierarchy_rollout_steps=16" \
  "training.hierarchy_batch_size=256" \
  "training.goal_energy_weight=0.0" \
  "+training.macro_action_advantage_weight=1.0" \
  "+training.macro_action_advantage_level=2" \
  "+training.macro_action_advantage_batch_size=256" \
  "+training.macro_action_advantage_standardize=true"

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

python -m puzzle_jepa.eval.diagnostics \
  --run-root "$RUN_ROOT" \
  --output-dir "$RUN_ROOT/diagnostics_subgoal_l2_macro_action_advantage" \
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
  --subgoal-high-score macro_action_advantage \
  --max-unroll-steps 81
