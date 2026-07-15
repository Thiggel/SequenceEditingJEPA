#!/usr/bin/env bash
#SBATCH --job-name=co_plan_iface
#SBATCH --partition=a40,rtxpro6k,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
unset VIRTUAL_ENV
export PATH="/usr/local/bin:/usr/bin:/bin"
unset XDG_CACHE_HOME HF_HOME HF_DATASETS_CACHE HUGGINGFACE_HUB_CACHE
unset TORCH_HOME TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR UV_CACHE_DIR
unset TMPDIR TMP TEMP
export PUZZLE_JEPA_WORK_ROOT="${PUZZLE_JEPA_WORK_ROOT:-/home/vault/$(id -gn)/${USER}/sequence-editing}"
export PUZZLE_JEPA_SCRATCH_ROOT="${PUZZLE_JEPA_WORK_ROOT}/scratch"
source scripts/env.sh

ROW="$(awk -F '\t' -v task_id="${SLURM_ARRAY_TASK_ID:?}" 'NR > 1 && $1 == task_id {print; exit}' "${TASK_MANIFEST:?}")"
IFS=$'\t' read -r _ objective seed mode hard_project reachability_weight <<<"${ROW}"
checkpoint="${CHECKPOINT_ROOT}/${objective}_s${seed}_joint_h110100/checkpoint.pt"
output_dir="${OUTPUT_ROOT}/${objective}_s${seed}"
mkdir -p "${output_dir}"
args=()
if [[ "${hard_project}" == true ]]; then
  args+=(--hard-project-support)
fi
python -m puzzle_jepa.train.controlled_objects_planning \
  "${checkpoint}" --output "${output_dir}/${mode}.json" --device auto \
  --seed "${PLANNING_SEED:-12017}" \
  --episodes "${PLANNING_EPISODES:-32}" \
  --candidates "${PLANNING_CANDIDATES:-512}" \
  --reachability-weight "${reachability_weight}" \
  --reachability-candidates "${REACHABILITY_CANDIDATES:-64}" \
  "${args[@]}"
