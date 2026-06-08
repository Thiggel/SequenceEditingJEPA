#!/usr/bin/env bash
#SBATCH --job-name=puzzle_grid4o_watch
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source ./scripts/env.sh

PROMPT_FILE="scripts/oversight/grid4o_one_shot_prompt.md"
if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "Missing prompt file: $PROMPT_FILE" >&2
  exit 2
fi

codex exec \
  --dangerously-bypass-approvals-and-sandbox \
  --cd "$PWD" \
  --add-dir "$PWD/../sequence-editing-report" \
  --output-last-message "logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_last_message.md" \
  - < "$PROMPT_FILE"
