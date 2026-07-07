#!/usr/bin/env bash
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p logs

VARIANTS=(raw_grid_energy proposal_energy jepa_energy)
IDS=()

for variant in "${VARIANTS[@]}"; do
  job_id="$(
    VARIANT="${variant}" \
      sbatch --parsable scripts/slurm/run_arc_jepa_train.slurm
  )"
  IDS+=("${variant}:${job_id}")
done

printf 'Submitted ARC first-pass training jobs:\n'
printf '  %s\n' "${IDS[@]}"
