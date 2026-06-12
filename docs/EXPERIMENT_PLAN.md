# Experiment Plan

Last updated: 2026-06-12 13:47 CEST

The active backlog lives in `../sequence-editing-report/BACKLOG.md`.

## Active: Grid 5 SIGReg Single-State JEPA

Goal: restart the Sudoku JEPA experiments around a compact single-state latent
with JEPA latent MSE plus SIGReg always enabled, and diagnose whether the latent
geometry is planner-ready.

Submitted Slurm array: `3722613_[0-23]`; completed cleanly with all tasks
exit `0:0`.

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

Grid 5 failed the gate. The reason is not training/Slurm failure: all variants
produced metrics and diagnostics. The failure is representational/planning:
oracle latent distances are mostly monotone along gold trajectories, but
all-action ranking is poor and planning solves `0/16` under both oracle
`latent_goal` and learned `goal_energy` for every variant.

The decisive diagnostics to inspect are:

- Does SIGReg produce approximately healthy latent spread?
- Is oracle latent-goal distance monotone along oracle trajectories?
- Does oracle latent-goal distance rank adjacent/action candidates correctly?
- Does the learned terminal-energy head track and rank the same candidates?
- Does small enumerated beam planning solve or at least reduce remaining
  Hamming under oracle latent distance?

Observed answer: monotone trajectories mostly pass, action ranking and planning
fail. The best oracle variant is `grid5_sigreg_mlp_mlp_delta_z128` with mean
remaining Hamming `44.88`, latent monotone rate `0.992`, latent gold-action
top-1 `0.031`, and latent top-goal-value rate `0.156`.

Next decision: do not spend on planner variants for this compact single-state
geometry until the action-ranking objective/representation is changed, or wait
for the still-running tokenized Grid 4Z control to decide whether tokenized
local geometry remains the better base.

## Active Posthoc: Grid 5 MPC-CEM Lookahead

Submitted as `3724325_[0-23]`.

Purpose: check whether the failed Grid 5 read was partly caused by the cheap
enumerated beam diagnostic rather than by the latent geometry. This follows the
LeWorldModel-style planning recipe more closely:

- CEM optimizes a sequence of symbolic Sudoku actions in latent space.
- Candidate sequences are rolled through the JEPA predictor.
- The final predicted latent is scored against the solved-board latent.
- MPC executes only the first action, updates the symbolic board, re-encodes,
  and replans.
- Horizons: `4`, `8`, `16`, `32`, `64`.

Artifacts:

- `diagnostics_mpc_cem/mpc_cem_summary.json`
- `diagnostics_mpc_cem/mpc_cem_records.jsonl`
- `diagnostics_mpc_cem/mpc_cem_root_actions.jsonl`
- `diagnostics_mpc_cem/mpc_cem_lookahead_examples.jsonl`

## Historical

Old `grid0`-`grid4` configs and Slurm wrappers were removed from the active
tree. Historical results remain in `../sequence-editing-report`.
