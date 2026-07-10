#!/usr/bin/env bash
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p logs

RUNS=(
  "semantic_mix_cls64_r1_base_prestage_lr1.0e-4_steps5000:3831210"
  "semantic_mix_cls64_r1_base_prestage_lr3.0e-4_steps5000:3831211"
  "semantic_mix_cls64_r1_base_prestage_lr1.0e-3_steps5000:3831212"
  "semantic_mix_cls64_r8_base_prestage_lr1.0e-4_steps5000:3831213"
  "semantic_mix_cls64_r8_base_prestage_lr3.0e-4_steps5000:3831214"
  "semantic_mix_cls64_r8_base_prestage_lr1.0e-3_steps5000:3831215"
  "semantic_mix_cls64_r1_ema_stability_lr1.0e-4_steps5000_seed1707:3831216"
  "semantic_mix_cls64_r1_ema_stability_lr3.0e-4_steps5000_seed1707:3831217"
  "semantic_mix_cls64_r1_vicreg_stability_lr1.0e-4_steps5000_seed1707:3831218"
  "semantic_mix_cls64_r1_vicreg_stability_lr3.0e-4_steps5000_seed1707:3831219"
  "semantic_mix_cls64_r1_sigreg_stability_lr1.0e-4_steps5000_seed1707:3831220"
  "semantic_mix_cls64_r1_sigreg_stability_lr3.0e-4_steps5000_seed1707:3831221"
  "semantic_mix_cls64_r8_ema_stability_lr1.0e-4_steps5000_seed1707:3831222"
  "semantic_mix_cls64_r8_ema_stability_lr3.0e-4_steps5000_seed1707:3831223"
  "semantic_mix_cls64_r8_vicreg_stability_lr1.0e-4_steps5000_seed1707:3831224"
  "semantic_mix_cls64_r8_vicreg_stability_lr3.0e-4_steps5000_seed1707:3831225"
  "semantic_mix_cls64_r8_sigreg_stability_lr1.0e-4_steps5000_seed1707:3831226"
  "semantic_mix_cls64_r8_sigreg_stability_lr3.0e-4_steps5000_seed1707:3831227"
)

IDS=()
for entry in "${RUNS[@]}"; do
  run_name="${entry%%:*}"
  train_job="${entry##*:}"
  if [[ "${SUBMIT:-0}" == "1" ]]; then
    train_state="$(sacct -j "${train_job}" -X -n -o State | awk 'NF {print $1; exit}')"
    dependency_args=()
    case "${train_state}" in
      COMPLETED*) ;;
      RUNNING*|PENDING*|CONFIGURING*|COMPLETING*) dependency_args=(--dependency="afterok:${train_job}") ;;
      *)
        printf 'Refusing re-probe for training job %s in state %s.\n' "${train_job}" "${train_state:-UNKNOWN}" >&2
        exit 2
        ;;
    esac
    job_id="$(
      RUN_NAME="${run_name}" \
        sbatch --parsable "${dependency_args[@]}" scripts/slurm/run_object_dynamics_probe_eval.slurm
    )"
    IDS+=("${run_name}:${job_id}")
  else
    printf 'DRY RUN: RUN_NAME=%s sbatch --dependency=afterok:%s scripts/slurm/run_object_dynamics_probe_eval.slurm\n' \
      "${run_name}" "${train_job}"
  fi
done

if [[ "${SUBMIT:-0}" == "1" ]]; then
  printf 'Submitted class-balanced object-dynamics re-probe jobs:\n'
  printf '  %s\n' "${IDS[@]}"
else
  printf 'Dry run only. Re-run with SUBMIT=1 to submit.\n'
fi
