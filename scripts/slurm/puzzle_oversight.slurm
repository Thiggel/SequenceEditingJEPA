#!/usr/bin/env bash
#SBATCH --job-name=puzzle_oversight
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-/home/hpc/c107fa/c107fa12/sequence-editing}"
cd "$REPO_ROOT"
mkdir -p logs
source ./scripts/env.sh

if [[ "${PUZZLE_OVERSIGHT_DISABLE_RESUBMIT:-0}" != "1" ]]; then
  if ! squeue -h -u "$USER" -n puzzle_oversight -t PD -o "%i" | grep -q .; then
    sbatch --begin=now+4hours "$REPO_ROOT/scripts/slurm/puzzle_oversight.slurm" || true
  fi
fi

shopt -s expand_aliases
if [[ -f "${HOME}/.bash_profile" ]]; then
  # shellcheck disable=SC1090
  source "${HOME}/.bash_profile"
elif [[ -f "${HOME}/.bashrc" ]]; then
  # shellcheck disable=SC1090
  source "${HOME}/.bashrc"
fi
hash -r 2>/dev/null || true

type cs >/dev/null 2>&1 || {
  echo "cs wrapper/alias not found after sourcing shell profile" >&2
  exit 127
}

PROMPT_PATH="$REPO_ROOT/scripts/oversight/puzzle_oversight_prompt.md"
if [[ ! -f "$PROMPT_PATH" ]]; then
  echo "Missing oversight prompt: $PROMPT_PATH" >&2
  exit 2
fi

cs exec "$(<"$PROMPT_PATH")"
