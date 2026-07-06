#!/usr/bin/env bash
#SBATCH --job-name=grid_goal_verifier_repair_train
#SBATCH --partition=rtxpro6k,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

VARIANTS=(
  B0_current_E4
  B1_current_F0
  S1_balanced_partials
  S2_near_solution
  S3_recovery_states
  S4_hard_action_sets
  W1_energy_x3
  W2_energy_x10
  W3_energy_x30
  W4_rank_x3
  W5_rank_x10
  G1_energy_projection
  G2_energy_finetune
  G3_dyn_half
  P1_pred_h1
  P2_pred_h4
  P3_pred_h8
  P4_score_consistency
  L1_pair_same_cell
  L2_pair_overwrite
  L3_listwise_32
  L4_listwise_128
  L5_exhaustive_small
  C1_sampling_weight
  C2_sampling_rank
  C3_pred_rank
  C4_energy_proj_rank
  C5_best_no_policy
  C6_best_policy
  C7_finetune_best
  C8_full
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

COMPATIBILITY_WEIGHT="1.0"
REMAINING_WEIGHT="1.0"
VERIFIER_PREDICTED_WEIGHT="0.5"
VERIFIER_PREDICTED_HORIZONS="[1]"
VERIFIER_CORRUPTION_WEIGHT="0.5"
VERIFIER_RANK_WEIGHT="0.0"
VERIFIER_RANK_MODE="none"
VERIFIER_ENERGY_PROJECTION="none"
POLICY_PRIOR_WEIGHT="0.0"
POLICY_PRIOR_PLANNING_WEIGHT="0.0"
POLICY_PRIOR_MODE="pairwise"
POLICY_PRIOR_TARGET="verifier"
CF_BRANCHES="16"
CF_DEPTH="4"
CF_MAX_PAIRS="128"
COUNTERFACTUAL_WEIGHT="1.0"
ORACLE_PROBABILITY="0.7"
EDITABLE_NOISE_PROBABILITY="0.15"
RANDOM_MAX_STEPS="96"
DENSE_FUTURE_WEIGHT="1.0"
LISTWISE_MAX_ACTIONS="128"
MAX_STEPS="${TRAIN_MAX_STEPS:-5000}"

case "${SELECTED_VARIANT}" in
  B0_current_E4)
    ;;
  B1_current_F0)
    VERIFIER_RANK_WEIGHT="1.0"; VERIFIER_RANK_MODE="listwise" ;;
  S1_balanced_partials)
    ORACLE_PROBABILITY="0.95"; EDITABLE_NOISE_PROBABILITY="0.05"; VERIFIER_CORRUPTION_WEIGHT="1.0" ;;
  S2_near_solution)
    ORACLE_PROBABILITY="1.0"; EDITABLE_NOISE_PROBABILITY="0.0"; VERIFIER_CORRUPTION_WEIGHT="3.0" ;;
  S3_recovery_states)
    ORACLE_PROBABILITY="0.6"; EDITABLE_NOISE_PROBABILITY="0.35"; RANDOM_MAX_STEPS="128"; VERIFIER_CORRUPTION_WEIGHT="1.0" ;;
  S4_hard_action_sets)
    VERIFIER_RANK_WEIGHT="1.0"; VERIFIER_RANK_MODE="listwise"; LISTWISE_MAX_ACTIONS="256"; CF_BRANCHES="32"; CF_MAX_PAIRS="256" ;;
  W1_energy_x3)
    COMPATIBILITY_WEIGHT="3.0"; REMAINING_WEIGHT="3.0" ;;
  W2_energy_x10)
    COMPATIBILITY_WEIGHT="10.0"; REMAINING_WEIGHT="10.0" ;;
  W3_energy_x30)
    COMPATIBILITY_WEIGHT="30.0"; REMAINING_WEIGHT="30.0" ;;
  W4_rank_x3)
    VERIFIER_RANK_WEIGHT="3.0"; VERIFIER_RANK_MODE="listwise" ;;
  W5_rank_x10)
    VERIFIER_RANK_WEIGHT="10.0"; VERIFIER_RANK_MODE="listwise" ;;
  G1_energy_projection)
    VERIFIER_ENERGY_PROJECTION="mlp" ;;
  G2_energy_finetune)
    COMPATIBILITY_WEIGHT="10.0"; REMAINING_WEIGHT="10.0"; VERIFIER_RANK_WEIGHT="3.0"; VERIFIER_RANK_MODE="listwise"; DENSE_FUTURE_WEIGHT="0.25"; MAX_STEPS="${TRAIN_MAX_STEPS:-3000}" ;;
  G3_dyn_half)
    DENSE_FUTURE_WEIGHT="0.5" ;;
  P1_pred_h1)
    VERIFIER_PREDICTED_HORIZONS="[1]" ;;
  P2_pred_h4)
    VERIFIER_PREDICTED_HORIZONS="[1,2,3,4]" ;;
  P3_pred_h8)
    VERIFIER_PREDICTED_HORIZONS="[1,2,3,4,5,6,7,8]" ;;
  P4_score_consistency)
    VERIFIER_PREDICTED_HORIZONS="[1,2,3,4,5,6,7,8]"; VERIFIER_PREDICTED_WEIGHT="1.5" ;;
  L1_pair_same_cell)
    VERIFIER_RANK_WEIGHT="3.0"; VERIFIER_RANK_MODE="pairwise" ;;
  L2_pair_overwrite)
    VERIFIER_RANK_WEIGHT="3.0"; VERIFIER_RANK_MODE="pairwise"; EDITABLE_NOISE_PROBABILITY="0.35" ;;
  L3_listwise_32)
    VERIFIER_RANK_WEIGHT="3.0"; VERIFIER_RANK_MODE="listwise"; LISTWISE_MAX_ACTIONS="32" ;;
  L4_listwise_128)
    VERIFIER_RANK_WEIGHT="3.0"; VERIFIER_RANK_MODE="listwise"; LISTWISE_MAX_ACTIONS="128" ;;
  L5_exhaustive_small)
    VERIFIER_RANK_WEIGHT="3.0"; VERIFIER_RANK_MODE="listwise"; LISTWISE_MAX_ACTIONS="729"; CF_BRANCHES="32"; CF_MAX_PAIRS="256" ;;
  C1_sampling_weight)
    COMPATIBILITY_WEIGHT="10.0"; REMAINING_WEIGHT="10.0"; VERIFIER_RANK_WEIGHT="1.0"; VERIFIER_RANK_MODE="listwise"; LISTWISE_MAX_ACTIONS="256"; CF_BRANCHES="32"; CF_MAX_PAIRS="256" ;;
  C2_sampling_rank)
    VERIFIER_RANK_WEIGHT="3.0"; VERIFIER_RANK_MODE="listwise"; LISTWISE_MAX_ACTIONS="256"; CF_BRANCHES="32"; CF_MAX_PAIRS="256" ;;
  C3_pred_rank)
    VERIFIER_PREDICTED_HORIZONS="[1,2,3,4,5,6,7,8]"; VERIFIER_PREDICTED_WEIGHT="1.0"; VERIFIER_RANK_WEIGHT="3.0"; VERIFIER_RANK_MODE="listwise"; LISTWISE_MAX_ACTIONS="128" ;;
  C4_energy_proj_rank)
    VERIFIER_ENERGY_PROJECTION="mlp"; VERIFIER_RANK_WEIGHT="3.0"; VERIFIER_RANK_MODE="listwise"; LISTWISE_MAX_ACTIONS="128" ;;
  C5_best_no_policy)
    COMPATIBILITY_WEIGHT="10.0"; REMAINING_WEIGHT="10.0"; VERIFIER_PREDICTED_HORIZONS="[1,2,3,4,5,6,7,8]"; VERIFIER_PREDICTED_WEIGHT="1.0"; VERIFIER_RANK_WEIGHT="3.0"; VERIFIER_RANK_MODE="listwise"; LISTWISE_MAX_ACTIONS="256"; CF_BRANCHES="32"; CF_MAX_PAIRS="256" ;;
  C6_best_policy)
    COMPATIBILITY_WEIGHT="10.0"; REMAINING_WEIGHT="10.0"; VERIFIER_PREDICTED_HORIZONS="[1,2,3,4,5,6,7,8]"; VERIFIER_PREDICTED_WEIGHT="1.0"; VERIFIER_RANK_WEIGHT="3.0"; VERIFIER_RANK_MODE="listwise"; LISTWISE_MAX_ACTIONS="256"; CF_BRANCHES="32"; CF_MAX_PAIRS="256"; POLICY_PRIOR_WEIGHT="0.5"; POLICY_PRIOR_PLANNING_WEIGHT="0.5"; POLICY_PRIOR_MODE="listwise" ;;
  C7_finetune_best)
    COMPATIBILITY_WEIGHT="10.0"; REMAINING_WEIGHT="10.0"; VERIFIER_PREDICTED_HORIZONS="[1,2,3,4,5,6,7,8]"; VERIFIER_PREDICTED_WEIGHT="1.0"; VERIFIER_RANK_WEIGHT="3.0"; VERIFIER_RANK_MODE="listwise"; LISTWISE_MAX_ACTIONS="256"; CF_BRANCHES="32"; CF_MAX_PAIRS="256"; DENSE_FUTURE_WEIGHT="0.25"; MAX_STEPS="${TRAIN_MAX_STEPS:-3000}" ;;
  C8_full)
    COMPATIBILITY_WEIGHT="10.0"; REMAINING_WEIGHT="10.0"; VERIFIER_PREDICTED_HORIZONS="[1,2,3,4,5,6,7,8]"; VERIFIER_PREDICTED_WEIGHT="1.0"; VERIFIER_CORRUPTION_WEIGHT="3.0"; VERIFIER_RANK_WEIGHT="3.0"; VERIFIER_RANK_MODE="listwise"; VERIFIER_ENERGY_PROJECTION="mlp"; LISTWISE_MAX_ACTIONS="256"; CF_BRANCHES="32"; CF_MAX_PAIRS="256"; ORACLE_PROBABILITY="0.8"; EDITABLE_NOISE_PROBABILITY="0.25"; POLICY_PRIOR_WEIGHT="0.5"; POLICY_PRIOR_PLANNING_WEIGHT="0.5"; POLICY_PRIOR_MODE="listwise" ;;
  *)
    echo "Unknown variant ${SELECTED_VARIANT}" >&2
    exit 2
    ;;
esac

RUN_SUFFIX="${RUN_SUFFIX:-_repair}"
RUN_NAME="grid_goal_verifier_repair_${SELECTED_VARIANT}${RUN_SUFFIX}"
OUTPUT_ROOT="${PUZZLE_JEPA_WORK_ROOT:-${WORK:?WORK must be set}/sequence-editing-repair-20260706}"
RUN_ROOT="${OUTPUT_ROOT}/runs/grid_goal_verifier_repair/${RUN_NAME}"

USER_OVERRIDES=()
if [[ -n "${EXTRA_HYDRA_OVERRIDES:-}" ]]; then
  read -r -a USER_OVERRIDES <<< "${EXTRA_HYDRA_OVERRIDES}"
fi

python -m puzzle_jepa.train.grid_goal_sudoku \
  --config-name grid_goal_sudoku \
  "seed=${SEED:-8801}" \
  "ablation=R4_no_goal_nce" \
  "+experiment_variant=${SELECTED_VARIANT}${RUN_SUFFIX}" \
  "output_dir=${RUN_ROOT}" \
  "task.train_limit=${TRAIN_LIMIT:-null}" \
  "task.eval_limit=${EVAL_LIMIT:-null}" \
  "training.max_steps=${MAX_STEPS}" \
  "training.batch_size=${BATCH_SIZE:-4}" \
  "training.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS:-2}" \
  "training.eval_every_steps=${EVAL_EVERY_STEPS:-500}" \
  "training.save_every_steps=${SAVE_EVERY_STEPS:-5000}" \
  "training.learning_rate=${LEARNING_RATE:-1.0e-4}" \
  "training.warmup_steps=${WARMUP_STEPS:-500}" \
  "training.bf16=${BF16:-true}" \
  "training.oracle_probability=${ORACLE_PROBABILITY}" \
  "training.allow_overwrite=true" \
  "training.editable_noise_probability=${EDITABLE_NOISE_PROBABILITY}" \
  "training.random_max_steps=${RANDOM_MAX_STEPS}" \
  "training.counterfactual_branches=${CF_BRANCHES}" \
  "training.counterfactual_depth=${CF_DEPTH}" \
  "training.counterfactual_max_pairs=${CF_MAX_PAIRS}" \
  "model.dropout=0.0" \
  "model.latent_representation=grid" \
  "model.action_conditioning=affected_marker" \
  "model.predict_delta=false" \
  "model.dynamics_target_mode=target_stopgrad" \
  "model.dynamics_weighting=affected_context" \
  "model.affected_dynamics_weight=${AFFECTED_DYNAMICS_WEIGHT:-8.0}" \
  "model.context_dynamics_weight=${CONTEXT_DYNAMICS_WEIGHT:-2.0}" \
  "model.regularizer=vicreg" \
  "model.sigreg_weight=0.1" \
  "model.use_ema_target_encoder=true" \
  "model.ema_decay=${EMA_DECAY:-0.995}" \
  "model.goal_mse_weight=0.0" \
  "model.goal_nce_weight=0.0" \
  "model.goal_distance_field_weight=0.0" \
  "model.waypoint_loss_weight=0.0" \
  "model.waypoint_final_weight=0.0" \
  "model.dense_future_weight=${DENSE_FUTURE_WEIGHT}" \
  "model.dense_rollout_all_steps=true" \
  "model.dense_rollout_variable_starts=false" \
  "model.dense_rollout_refactor_mode=none" \
  "model.dense_rollout_weighting=smooth_count" \
  "model.dense_rollout_gamma=${DENSE_ROLLOUT_GAMMA:-0.8}" \
  "model.multi_step_horizons=[8]" \
  "model.rollout_detach_interval=0" \
  "model.counterfactual_dynamics_weight=${COUNTERFACTUAL_WEIGHT}" \
  "model.hierarchy_levels=[]" \
  "model.hierarchy_loss_weight=0.0" \
  "model.hierarchy_dense_future_weight=0.0" \
  "model.progress_rank_target=none" \
  "model.progress_rank_weight=0.0" \
  "model.action_rank_mode=none" \
  "model.action_rank_weight=0.0" \
  "model.temporal_straightening_weight=0.0" \
  "model.terminal_corrupt_weight=0.0" \
  "model.compatibility_weight=${COMPATIBILITY_WEIGHT}" \
  "model.remaining_weight=${REMAINING_WEIGHT}" \
  "model.verifier_predicted_weight=${VERIFIER_PREDICTED_WEIGHT}" \
  "model.verifier_predicted_horizons=${VERIFIER_PREDICTED_HORIZONS}" \
  "model.verifier_corruption_weight=${VERIFIER_CORRUPTION_WEIGHT}" \
  "model.verifier_rank_weight=${VERIFIER_RANK_WEIGHT}" \
  "model.verifier_rank_mode=${VERIFIER_RANK_MODE}" \
  "model.verifier_energy_projection=${VERIFIER_ENERGY_PROJECTION}" \
  "model.verifier_score_alpha=${VERIFIER_SCORE_ALPHA:-1.0}" \
  "model.verifier_score_beta=${VERIFIER_SCORE_BETA:-1.0}" \
  "model.policy_prior_weight=${POLICY_PRIOR_WEIGHT}" \
  "model.policy_prior_mode=${POLICY_PRIOR_MODE}" \
  "model.policy_prior_planning_weight=${POLICY_PRIOR_PLANNING_WEIGHT}" \
  "model.policy_prior_target=${POLICY_PRIOR_TARGET}" \
  "model.listwise_action_rank_max_actions=${LISTWISE_MAX_ACTIONS}" \
  "${USER_OVERRIDES[@]}"
