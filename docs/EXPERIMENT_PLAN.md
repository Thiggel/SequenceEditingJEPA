# Experiment Plan

Last updated: 2026-06-12 11:30 CEST

The active backlog lives in `../sequence-editing-report/BACKLOG.md`.

## Active: Grid 5 SIGReg Single-State JEPA

Goal: restart the Sudoku JEPA experiments around a compact single-state latent
with JEPA latent MSE plus SIGReg always enabled, and diagnose whether the latent
geometry is planner-ready.

Submitted Slurm array: `3722613_[0-23]`

| Factor | Values |
| --- | --- |
| Encoder | one-hidden-layer board `mlp`; bidirectional `cls_transformer` |
| Predictor | one-hidden-layer `mlp`; causal `ar_transformer` over state/action sequence |
| Prediction target | full next latent; residual delta |
| Latent size | `32`, `64`, `128` |

Fixed defaults:

- `sigreg_weight=1.0`
- terminal-energy head enabled and trained in every model
- `action_size=16`
- rollout length `8`
- rollout training mix `50%` oracle/correct and `50%` wrong/random mutable
  trajectories
- batch size `512`, max steps `5000`

## Gate

Do not judge Grid 5 only by solve rate. First check:

- Does SIGReg produce approximately healthy latent spread?
- Is oracle latent-goal distance monotone along oracle trajectories?
- Does oracle latent-goal distance rank adjacent/action candidates correctly?
- Does the learned terminal-energy head track and rank the same candidates?
- Does small enumerated beam planning solve or at least reduce remaining
  Hamming under oracle latent distance?

If oracle latent geometry fails, change representation/objective before planner
work. If oracle latent geometry passes but learned energy fails, focus on the
terminal-energy objective. If both pass but planning fails, revisit planner
horizon/beam/CEM.

## Historical

Old `grid0`-`grid4` configs and Slurm wrappers were removed from the active
tree. Historical results remain in `../sequence-editing-report`.
