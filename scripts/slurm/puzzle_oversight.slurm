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
cd "${SLURM_SUBMIT_DIR:-/home/hpc/c107fa/c107fa12/sequence-editing}"
mkdir -p logs
source ./scripts/env.sh

if [[ "${PUZZLE_OVERSIGHT_DISABLE_RESUBMIT:-0}" != "1" ]]; then
  sbatch --begin=now+4hours "$0" || true
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

cs exec "You are running scheduled autonomous oversight for /home/hpc/c107fa/c107fa12/sequence-editing.

Read AGENTS.md, docs/RUNBOOK.md, docs/RESULTS.md, docs/EXPERIMENT_PLAN.md, and legacy.md first. Keep the handoff docs current in this same turn when jobs finish, fail, get submitted, or new insights are found.

Use source scripts/env.sh for shell commands. Do not change package versions. Do not modify ../legacy-sequence-editing except to inspect archived legacy files if needed.

Tasks:
- Check Slurm with squeue and recent sacct for puzzle_grid0, puzzle_grid1, puzzle_grid2a, puzzle_grid3a, puzzle_grid3b, puzzle_diag, puzzle_diag2a, puzzle_diag3a, puzzle_diag3b_large, puzzle_diag3b_n2, and puzzle_oversight jobs.
- Inspect logs/, Grid 0 roots, Grid 1 roots, Grid 2A rollout roots, Grid 3A local-edit roots, and any diagnostics outputs under PUZZLE_JEPA_WORK_ROOT/runs.
- If Grid 0 jobs are still running and making progress, report status and leave them alone.
- If a Grid 0 job failed, timed out, OOMed, or produced no checkpoint/metrics, inspect the traceback. If the fix is small and local, implement it, run focused pytest/smoke checks, update docs, and resubmit only the failed Grid 0 job or a safer config. For OOM, lower batch size first.
- If both Grid 0 jobs completed and metrics.json plus checkpoint.pt exist for both, parse metrics, update docs/RUNBOOK.md, docs/RESULTS.md, docs/EXPERIMENT_PLAN.md, and submit Grid 1 only if the pipeline pass conditions in docs/EXPERIMENT_PLAN.md are met. If Grid 1 scripts/configs are not implemented yet, implement the minimal configs and Slurm wrapper for the documented Grid 1 data curriculum, test locally, and submit.
- If Grid 1 is running and making progress, report status and leave it alone. If a Grid 1 task fails, inspect the traceback and resubmit only the failed task or a safer config after a focused fix.
- If puzzle_diag is running and making progress, report status and leave it alone. If diagnostics finish, parse diagnostics.json files, update handoff docs with drift/rank/terminal-energy insights, and only suggest the next experiment if the diagnostics identify a concrete bottleneck. If a diagnostics task fails, inspect the traceback, make the smallest local fix, run focused tests/smokes, update docs, and resubmit only the failed diagnostics task or a safer config.
- If Grid 2A rollout jobs are running and making progress, report status and leave them alone. If a Grid 2A task fails, inspect the traceback and resubmit only the failed task or a safer config after a focused fix. For OOM, lower rollout_batch_size before lowering one-step batch size.
- If all Grid 2A rollout jobs complete and their checkpoints/metrics exist, ensure dependent Grid 2A diagnostics have been submitted. If Grid 2A diagnostics finish, parse diagnostics.json files and compare their 10/20/terminal drift, action-rank curves, and step/terminal planning against Grid 1 diagnostics. Update handoff docs with whether N=2/N=4 rollout training reduced drift and whether 10M N=1/N=2 controls are now warranted.
- If Grid 3A local-edit jobs are running and making progress, report status and leave them alone. If a Grid 3A task fails, inspect the traceback and resubmit only the failed task or a safer config after a focused fix. For OOM, lower batch_size first.
- If all Grid 3A jobs complete and their checkpoints/metrics exist, ensure dependent Grid 3A diagnostics have been submitted. If Grid 3A diagnostics finish, parse diagnostics.json files, including goal_rank if present, and compare local direct uniform, local direct weighted, local residual weighted, and local direct changed-only against Sudoku Grid 1/2A. Update handoff docs with whether local action injection, loss weighting, or residual prediction improved action grounding or drift.
- If Grid 3B large diagnostics for the Grid 3A lead are running, leave them alone unless they fail. If they finish, parse diagnostics_large/diagnostics.json plus latent_planning_records.jsonl and reencoded_planning_records.jsonl; report whether re-encoded planning solves more boards than latent rollout and where terminal board errors concentrate.
- If Grid 3B rollout N=2 is running, leave it alone unless it fails. If it completes, ensure dependent Grid 3B N=2 diagnostics have been submitted. If those diagnostics finish, compare N=2 against Grid 3A local_direct_weighted on goal_rank, 10/20/terminal drift, latent-vs-reencoded planning, terminal solve, and remaining Hamming.
- Ensure this oversight chain continues every 4 hours. If no later puzzle_oversight job is pending, submit scripts/slurm/puzzle_oversight.slurm with --begin=now+4hours.
- End with concise status: job IDs, states, output roots, checkpoints, latest metrics or failure reasons, docs updated, and next safe steps."
