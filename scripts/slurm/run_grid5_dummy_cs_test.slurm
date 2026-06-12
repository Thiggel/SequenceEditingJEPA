#!/bin/bash
#SBATCH --job-name=grid5_dummy_cs
#SBATCH --output=logs/grid5_dummy_cs_%j.out
#SBATCH --error=logs/grid5_dummy_cs_%j.err
#SBATCH --partition=a40,a100,rtxpro6k
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:05:00

set -euo pipefail

cd /home/hpc/c107fa/c107fa12/sequence-editing
source ./scripts/env.sh
mkdir -p logs

echo "Testing shell alias/function 'cs' from ~/.bash_profile."

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
  "$(cat scripts/oversight/grid5_dummy_prompt.md)"
