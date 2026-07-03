# Current Experiments

Source of truth: `../sequence-editing-report/CURRENT_EXPERIMENTS.md`.

## Prepared: Counterfactual Editable Weekend Wave

Status: implemented and submit-ready; not submitted yet.

The audit blockers are fixed:

- counterfactual branches now store P-step future targets from the branch root
  plus explicit action/future-board sequences
- counterfactual sequences supervise primitive dynamics and hierarchy chunks
  when sampled depth covers the hierarchy level
- multi-horizon waypoint configs produce one latent per horizon
- hierarchical beam preserves editable overwrite mode into primitive tracking
- oversight defaults to 12-hour checks and mechanical eval repair is enabled
  by default

Important invariant: every Delta-JEPA experiment has both a full-grid latent
run and a single learned-CLS latent run. Oversight follow-ups must preserve
that pair.

Prepared scripts:

- `scripts/experiments/submit_grid_goal_weekend.sh`
- `scripts/experiments/submit_grid_goal_weekend_oversight.sh`
- `scripts/slurm/run_grid_goal_weekend_train.slurm`
- `scripts/slurm/run_grid_goal_weekend_eval.slurm`
- `scripts/slurm/run_grid_goal_weekend_oversight.slurm`
- `scripts/experiments/grid_goal_weekend_manifest.json`

The submit wrapper will launch one training array with 30 variants, then 30
independent dependency-held eval jobs, one per train array task.
