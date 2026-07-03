# Current Experiments

Source of truth: `../sequence-editing-report/CURRENT_EXPERIMENTS.md`.

## Prepared: Counterfactual Editable Weekend Wave

Status: implemented but blocked by audit fixes; not submitted yet.

This sweep tests whether the main failures were weak action dependence,
irreversible fill-only Sudoku dynamics, and one-shot terminal goal prediction.
The prepared jobs add counterfactual branch supervision, editable non-clue
cells, receding-horizon waypoint prediction, asymmetric/value metric variants,
and a paired Delta-JEPA branch.

Do not submit this grid until the audit blockers are fixed:

- counterfactual depth must produce P-step branch targets rather than only
  chained one-step pairs
- counterfactual branches must train upper hierarchy levels
- multi-horizon waypoint configs need separate waypoint outputs
- hierarchical beam must preserve editable overwrite mode into primitive
  tracking
- weekend oversight should default to 12-hour mechanical checks with repair
  enabled for failed/missing evals

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
