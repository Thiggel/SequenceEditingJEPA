# Results

Last updated: 2026-06-14

## Current Result

No clean LeWorldModel reset jobs have completed yet. The latest pre-sweep review
regressions are fixed and verified, but no replacement job has been submitted.

Cancelled/superseded job `3740707_[0-24%12]` should not be used as the clean
baseline. It trained with 8-frame trajectories while planning included horizons
up to 64, and its MCTS implementation did not perform meaningful tree search at
Sudoku root branching scale. No replacement job has been submitted.

All previous Grid4-Grid6 experiments are legacy context. The short read is:
older tokenized oracle-goal reset controls could solve Sudoku when given the
solved board latent, but learned scalar goal-energy/value heads and compact
single-state variants failed to rank actions or terminal boards reliably. This
reset removes the old experiment surface and reruns the clean LeWM recipe.

## New Gate

The LR sweep must answer:

- Does step-wise LeWM SIGReg produce a healthy Gaussian-like latent spread?
- Is oracle latent distance to the solved board monotone on true fill-only
  trajectories?
- Do oracle latent distance and the learned goal-distance head rank local
  candidate actions correctly?
- Which MPC inner planner works best for Sudoku under fill-only actions:
  greedy, beam, best-first/weighted A*, categorical CEM, local search, or UCT
  MCTS?
- Does symbolic re-encode outperform latent rollout, and at which horizons
  `4/8/16/32/64`?

Every run writes diagnostics and a planner matrix under its run root.

Current code trains full correct/wrong fill-only trajectories with masks, uses
LeWM-style MLP projectors, keeps padded frames out of BatchNorm projector
statistics, uses full-history latent rollout during MPC, and reports default
MCTS as score-pruned progressive UCT. The latest review fixes also make
predictor projection BatchNorm step-wise, remove post-AdaLN sublayer
normalization, keep state embeddings independent of the `goals` argument, cap
latent-rollout lookahead to the available predictor context, and pass history
into branch-pruned latent rollout plus projection-panel latent-rollout
diagnostics.
Diagnostics now also include train-vs-eval terminal goal-distance checks,
predictor full-vs-truncated BatchNorm deltas, no-history vs full-history latent
rank divergence, branch-prune gold-action survival, latent-rollout drift
against symbolic re-encode by horizon, and planner timing/action-eval counts.

Verification before resubmission:

- `pytest -q` passes.
- Tiny train smoke writes scalar metrics plus `diagnostics/` JSONL/CSV/SVG
  artifacts.
- Standalone planner-matrix CLI smoke runs from the smoke checkpoint.
