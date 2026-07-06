#!/usr/bin/env bash
#SBATCH --job-name=grid_goal_verifier_train
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
  E0_base_oracle_sanity
  E1_compat_state
  E2_remaining_state
  E3_wr_state
  E4_wr_predicted
  E5_wr_pairwise_rank
  E6_wr_listwise_rank
  E7_wr_listwise_policy
  E8_wr_no_counterfactual
  E9_wr_no_corruption
  F0_full_score
  F1_full_policy
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

COMPATIBILITY_WEIGHT="0.0"
REMAINING_WEIGHT="0.0"
VERIFIER_PREDICTED_WEIGHT="0.0"
VERIFIER_CORRUPTION_WEIGHT="0.0"
VERIFIER_RANK_WEIGHT="0.0"
VERIFIER_RANK_MODE="pairwise"
POLICY_PRIOR_WEIGHT="0.0"
POLICY_PRIOR_PLANNING_WEIGHT="0.0"
POLICY_PRIOR_MODE="pairwise"
POLICY_PRIOR_TARGET="goal"
CF_BRANCHES="16"
CF_DEPTH="4"
CF_MAX_PAIRS="128"
COUNTERFACTUAL_WEIGHT="1.0"

case "${SELECTED_VARIANT}" in
  E0_base_oracle_sanity)
    ;;
  E1_compat_state)
    COMPATIBILITY_WEIGHT="1.0"; VERIFIER_CORRUPTION_WEIGHT="0.5" ;;
  E2_remaining_state)
    REMAINING_WEIGHT="1.0"; VERIFIER_CORRUPTION_WEIGHT="0.5" ;;
  E3_wr_state)
    COMPATIBILITY_WEIGHT="1.0"; REMAINING_WEIGHT="1.0"; VERIFIER_CORRUPTION_WEIGHT="0.5" ;;
  E4_wr_predicted)
    COMPATIBILITY_WEIGHT="1.0"; REMAINING_WEIGHT="1.0"; VERIFIER_PREDICTED_WEIGHT="0.5"; VERIFIER_CORRUPTION_WEIGHT="0.5" ;;
  E5_wr_pairwise_rank)
    COMPATIBILITY_WEIGHT="1.0"; REMAINING_WEIGHT="1.0"; VERIFIER_PREDICTED_WEIGHT="0.5"; VERIFIER_CORRUPTION_WEIGHT="0.5"; VERIFIER_RANK_WEIGHT="1.0"; VERIFIER_RANK_MODE="pairwise" ;;
  E6_wr_listwise_rank|F0_full_score)
    COMPATIBILITY_WEIGHT="1.0"; REMAINING_WEIGHT="1.0"; VERIFIER_PREDICTED_WEIGHT="0.5"; VERIFIER_CORRUPTION_WEIGHT="0.5"; VERIFIER_RANK_WEIGHT="1.0"; VERIFIER_RANK_MODE="listwise" ;;
  E7_wr_listwise_policy|F1_full_policy)
    COMPATIBILITY_WEIGHT="1.0"; REMAINING_WEIGHT="1.0"; VERIFIER_PREDICTED_WEIGHT="0.5"; VERIFIER_CORRUPTION_WEIGHT="0.5"; VERIFIER_RANK_WEIGHT="1.0"; VERIFIER_RANK_MODE="listwise"; POLICY_PRIOR_WEIGHT="0.2"; POLICY_PRIOR_PLANNING_WEIGHT="0.2"; POLICY_PRIOR_MODE="listwise"; POLICY_PRIOR_TARGET="verifier" ;;
  E8_wr_no_counterfactual)
    COMPATIBILITY_WEIGHT="1.0"; REMAINING_WEIGHT="1.0"; VERIFIER_PREDICTED_WEIGHT="0.5"; VERIFIER_CORRUPTION_WEIGHT="0.5"; VERIFIER_RANK_WEIGHT="1.0"; VERIFIER_RANK_MODE="listwise"; CF_BRANCHES="0"; CF_MAX_PAIRS="0"; COUNTERFACTUAL_WEIGHT="0.0" ;;
  E9_wr_no_corruption)
    COMPATIBILITY_WEIGHT="1.0"; REMAINING_WEIGHT="1.0"; VERIFIER_PREDICTED_WEIGHT="0.5"; VERIFIER_CORRUPTION_WEIGHT="0.0"; VERIFIER_RANK_WEIGHT="1.0"; VERIFIER_RANK_MODE="listwise" ;;
  *)
    echo "Unknown variant ${SELECTED_VARIANT}" >&2
    exit 2
    ;;
esac

RUN_SUFFIX="${RUN_SUFFIX:-}"
RUN_NAME="grid_goal_verifier_energy_${SELECTED_VARIANT}${RUN_SUFFIX}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/grid_goal_verifier_energy/${RUN_NAME}"

USER_OVERRIDES=()
if [[ -n "${EXTRA_HYDRA_OVERRIDES:-}" ]]; then
  read -r -a USER_OVERRIDES <<< "${EXTRA_HYDRA_OVERRIDES}"
fi

python -m puzzle_jepa.train.grid_goal_sudoku \
  --config-name grid_goal_sudoku \
  "seed=${SEED:-7801}" \
  "ablation=R4_no_goal_nce" \
  "+experiment_variant=${SELECTED_VARIANT}${RUN_SUFFIX}" \
  "output_dir=${RUN_ROOT}" \
  "task.train_limit=${TRAIN_LIMIT:-null}" \
  "task.eval_limit=${EVAL_LIMIT:-null}" \
  "training.max_steps=${TRAIN_MAX_STEPS:-5000}" \
  "training.batch_size=${BATCH_SIZE:-4}" \
  "training.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS:-2}" \
  "training.eval_every_steps=${EVAL_EVERY_STEPS:-500}" \
  "training.save_every_steps=${SAVE_EVERY_STEPS:-5000}" \
  "training.learning_rate=${LEARNING_RATE:-1.0e-4}" \
  "training.warmup_steps=${WARMUP_STEPS:-500}" \
  "training.bf16=${BF16:-true}" \
  "training.oracle_probability=${ORACLE_PROBABILITY:-0.7}" \
  "training.allow_overwrite=true" \
  "training.editable_noise_probability=${EDITABLE_NOISE_PROBABILITY:-0.15}" \
  "training.random_max_steps=${RANDOM_MAX_STEPS:-96}" \
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
  "model.dense_future_weight=1.0" \
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
  "model.verifier_corruption_weight=${VERIFIER_CORRUPTION_WEIGHT}" \
  "model.verifier_rank_weight=${VERIFIER_RANK_WEIGHT}" \
  "model.verifier_rank_mode=${VERIFIER_RANK_MODE}" \
  "model.verifier_score_alpha=${VERIFIER_SCORE_ALPHA:-1.0}" \
  "model.verifier_score_beta=${VERIFIER_SCORE_BETA:-1.0}" \
  "model.policy_prior_weight=${POLICY_PRIOR_WEIGHT}" \
  "model.policy_prior_mode=${POLICY_PRIOR_MODE}" \
  "model.policy_prior_planning_weight=${POLICY_PRIOR_PLANNING_WEIGHT}" \
  "model.policy_prior_target=${POLICY_PRIOR_TARGET}" \
  "model.listwise_action_rank_max_actions=${LISTWISE_ACTION_RANK_MAX_ACTIONS:-128}" \
  "${USER_OVERRIDES[@]}"
