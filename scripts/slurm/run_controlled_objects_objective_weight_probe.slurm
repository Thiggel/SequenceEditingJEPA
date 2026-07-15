#!/usr/bin/env bash
#SBATCH --job-name=co_obj_probe
#SBATCH --partition=a40,rtxpro6k,a100,a100mig
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
IFS=$'\t' read -r _ variant _ _ _ _ seed <<<"${ROW}"
run_name="${variant}_s${seed}_joint_h110100"
python -m puzzle_jepa.train.controlled_objects_probes \
  "${OUTPUT_ROOT}/${run_name}/checkpoint.pt" \
  --output "${OUTPUT_ROOT}/${run_name}/probe_eval_v5.json" \
  --device auto --probe-seed 9917 \
  --train-samples "${PROBE_TRAIN_SAMPLES:-1024}" \
  --eval-samples "${PROBE_EVAL_SAMPLES:-512}" \
  --batch-size "${PROBE_BATCH_SIZE:-64}" \
  --steps "${PROBE_STEPS:-500}" --learning-rate 0.003 \
  --planning-episodes 0 --planning-candidates 512
