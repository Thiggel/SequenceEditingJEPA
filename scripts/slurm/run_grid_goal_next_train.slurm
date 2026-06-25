#!/usr/bin/env bash
#SBATCH --job-name=grid_goal_next_train
#SBATCH --partition=rtxpro6k,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-6%4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

STAGE="${GRID_GOAL_STAGE:-goal_conditioning}"
INDEX="${SLURM_ARRAY_TASK_ID:-0}"

case "${STAGE}" in
  goal_conditioning)
    VARIANTS=(G0_context G1_initial_current G2_initial_current_oracle_progress)
    ;;
  dense_horizon)
    VARIANTS=(DK2 DK4 DK8 DK16 DK32)
    ;;
  hierarchy_levels)
    VARIANTS=(H_empty H2 H2_4 H2_4_8 H2_4_8_16 H2_4_8_16_32)
    ;;
  predictor_delta_topk)
    VARIANTS=(P0_separate_delta P1_shared_delta P2_shared_no_delta P3_shared_delta_topk_train)
    ;;
  ranking_losses)
    VARIANTS=(R_pred_progress R_oracle_progress R_both_progress R_no_progress R_pairwise R_listwise R_no_action_rank)
    ;;
  hierarchical_planning)
    VARIANTS=(BASE)
    ;;
  policy_prior)
    VARIANTS=(PP0_no_prior PP1_pairwise_prior PP2_listwise_prior PP3_listwise_prior_strong)
    ;;
  *)
    echo "Unknown GRID_GOAL_STAGE=${STAGE}" >&2
    exit 2
    ;;
esac

if (( INDEX < 0 || INDEX >= ${#VARIANTS[@]} )); then
  echo "Skipping array index ${INDEX} for stage ${STAGE}; variant count is ${#VARIANTS[@]}."
  exit 0
fi

VARIANT="${VARIANTS[$INDEX]}"
RUN_NAME="grid_goal_next_${STAGE}_${VARIANT}"
RUN_ROOT="${PUZZLE_JEPA_WORK_ROOT}/runs/grid_goal_next_wave/${RUN_NAME}"
SEED=$(( 6201 + INDEX ))

COMMON_OVERRIDES=(
  --config-name grid_goal_sudoku
  "seed=${SEED}"
  "ablation=R4_no_goal_nce"
  "output_dir=${RUN_ROOT}"
  "task.train_limit=${TRAIN_LIMIT:-null}"
  "task.eval_limit=${EVAL_LIMIT:-null}"
  "training.max_steps=${TRAIN_MAX_STEPS:-45000}"
  "training.batch_size=${BATCH_SIZE:-4}"
  "training.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS:-1}"
  "training.eval_every_steps=${EVAL_EVERY_STEPS:-1000}"
  "training.save_every_steps=${SAVE_EVERY_STEPS:-5000}"
  "training.learning_rate=${LEARNING_RATE:-1.0e-4}"
  "model.action_conditioning=affected_marker"
  "model.predict_delta=true"
  "model.dynamics_weighting=uniform"
  "model.regularizer=vicreg"
  "model.use_ema_target_encoder=true"
  "model.ema_decay=${EMA_DECAY:-0.995}"
  "model.temporal_straightening_weight=${TEMPORAL_STRAIGHTENING_WEIGHT:-0.1}"
  "model.progress_rank_weight=${PROGRESS_RANK_WEIGHT:-1.0}"
  "model.progress_rank_target=predicted"
  "model.goal_conditioning=initial_current"
  "model.dense_future_weight=${DENSE_FUTURE_WEIGHT:-1.0}"
)

EXTRA_OVERRIDES=()
case "${VARIANT}" in
  G0_context)
    EXTRA_OVERRIDES=("model.goal_conditioning=context")
    ;;
  G1_initial_current)
    EXTRA_OVERRIDES=("model.goal_conditioning=initial_current")
    ;;
  G2_initial_current_oracle_progress)
    EXTRA_OVERRIDES=("model.goal_conditioning=initial_current" "model.progress_rank_target=both")
    ;;
  DK*)
    K="${VARIANT#DK}"
    EXTRA_OVERRIDES=("model.multi_step_horizons=[1,${K}]" "model.hierarchy_levels=[2,4,8,16]" "model.hierarchy_loss_weight=1.0")
    ;;
  H_empty)
    EXTRA_OVERRIDES=("model.hierarchy_levels=[]" "model.hierarchy_loss_weight=0.0" "model.multi_step_horizons=[1,4,8,16]")
    ;;
  H2)
    EXTRA_OVERRIDES=("model.hierarchy_levels=[2]" "model.hierarchy_loss_weight=1.0" "model.multi_step_horizons=[1,4,8,16]")
    ;;
  H2_4)
    EXTRA_OVERRIDES=("model.hierarchy_levels=[2,4]" "model.hierarchy_loss_weight=1.0" "model.multi_step_horizons=[1,4,8,16]")
    ;;
  H2_4_8)
    EXTRA_OVERRIDES=("model.hierarchy_levels=[2,4,8]" "model.hierarchy_loss_weight=1.0" "model.multi_step_horizons=[1,4,8,16]")
    ;;
  H2_4_8_16)
    EXTRA_OVERRIDES=("model.hierarchy_levels=[2,4,8,16]" "model.hierarchy_loss_weight=1.0" "model.multi_step_horizons=[1,4,8,16]")
    ;;
  H2_4_8_16_32)
    EXTRA_OVERRIDES=("model.hierarchy_levels=[2,4,8,16,32]" "model.hierarchy_loss_weight=1.0" "model.multi_step_horizons=[1,4,8,16,32]")
    ;;
  P0_separate_delta)
    EXTRA_OVERRIDES=("model.hierarchy_levels=[2,4,8,16]" "model.hierarchy_loss_weight=1.0" "model.shared_hierarchy_predictor=false" "model.predict_delta=true")
    ;;
  P1_shared_delta)
    EXTRA_OVERRIDES=("model.hierarchy_levels=[2,4,8,16]" "model.hierarchy_loss_weight=1.0" "model.shared_hierarchy_predictor=true" "model.predict_delta=true")
    ;;
  P2_shared_no_delta)
    EXTRA_OVERRIDES=("model.hierarchy_levels=[2,4,8,16]" "model.hierarchy_loss_weight=1.0" "model.shared_hierarchy_predictor=true" "model.predict_delta=false")
    ;;
  P3_shared_delta_topk_train)
    EXTRA_OVERRIDES=("model.hierarchy_levels=[2,4,8,16]" "model.hierarchy_loss_weight=1.0" "model.shared_hierarchy_predictor=true" "model.predict_delta=true" "model.dynamics_weighting=affected")
    ;;
  R_pred_progress)
    EXTRA_OVERRIDES=("model.progress_rank_target=predicted")
    ;;
  R_oracle_progress)
    EXTRA_OVERRIDES=("model.progress_rank_target=oracle")
    ;;
  R_both_progress)
    EXTRA_OVERRIDES=("model.progress_rank_target=both")
    ;;
  R_no_progress)
    EXTRA_OVERRIDES=("model.progress_rank_target=none" "model.progress_rank_weight=0.0")
    ;;
  R_pairwise)
    EXTRA_OVERRIDES=("model.action_rank_mode=pairwise")
    ;;
  R_listwise)
    EXTRA_OVERRIDES=("model.action_rank_mode=listwise" "model.listwise_action_rank_max_actions=${LISTWISE_ACTION_RANK_MAX_ACTIONS:-128}")
    ;;
  R_no_action_rank)
    EXTRA_OVERRIDES=("model.action_rank_mode=none" "model.action_rank_weight=0.0")
    ;;
  BASE)
    EXTRA_OVERRIDES=("model.hierarchy_levels=[2,4,8,16]" "model.hierarchy_loss_weight=1.0")
    ;;
  PP0_no_prior)
    EXTRA_OVERRIDES=("model.hierarchy_levels=[2,4,8,16]" "model.hierarchy_loss_weight=1.0" "model.shared_hierarchy_predictor=true" "model.policy_prior_weight=0.0" "model.policy_prior_planning_weight=0.0")
    ;;
  PP1_pairwise_prior)
    EXTRA_OVERRIDES=("model.hierarchy_levels=[2,4,8,16]" "model.hierarchy_loss_weight=1.0" "model.shared_hierarchy_predictor=true" "model.policy_prior_mode=pairwise" "model.policy_prior_weight=0.5" "model.policy_prior_planning_weight=0.25")
    ;;
  PP2_listwise_prior)
    EXTRA_OVERRIDES=("model.hierarchy_levels=[2,4,8,16]" "model.hierarchy_loss_weight=1.0" "model.shared_hierarchy_predictor=true" "model.policy_prior_mode=listwise" "model.listwise_action_rank_max_actions=${LISTWISE_ACTION_RANK_MAX_ACTIONS:-128}" "model.policy_prior_weight=0.5" "model.policy_prior_planning_weight=0.25")
    ;;
  PP3_listwise_prior_strong)
    EXTRA_OVERRIDES=("model.hierarchy_levels=[2,4,8,16]" "model.hierarchy_loss_weight=1.0" "model.shared_hierarchy_predictor=true" "model.policy_prior_mode=listwise" "model.listwise_action_rank_max_actions=${LISTWISE_ACTION_RANK_MAX_ACTIONS:-128}" "model.policy_prior_weight=1.0" "model.policy_prior_planning_weight=0.5")
    ;;
esac

python -m puzzle_jepa.train.grid_goal_sudoku "${COMMON_OVERRIDES[@]}" "${EXTRA_OVERRIDES[@]}"
