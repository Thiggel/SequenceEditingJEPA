#!/bin/bash
#SBATCH --job-name=grid5_watch
#SBATCH --output=logs/grid5_watch_%j.out
#SBATCH --error=logs/grid5_watch_%j.err
#SBATCH --partition=a40,a100,rtxpro6k
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

set -euo pipefail

cd /home/hpc/c107fa/c107fa12/sequence-editing
source ./scripts/env.sh
mkdir -p logs

PROMPT_FILE="${GRID5_OVERSIGHT_PROMPT:-scripts/oversight/grid5_oversight_prompt.md}"
LAST_MESSAGE="logs/grid5_watch_${SLURM_JOB_ID:-manual}.last_message.txt"

shopt -s expand_aliases
# shellcheck disable=SC1090
source "$HOME/.bash_profile"
type cs

cs \
  -C /home/hpc/c107fa/c107fa12/sequence-editing \
  --add-dir /home/hpc/c107fa/c107fa12/sequence-editing-report \
  -c 'model_reasoning_effort="medium"' \
  -c shell_environment_policy.inherit=all \
  exec \
  -o "${LAST_MESSAGE}" \
  - < "${PROMPT_FILE}"
