# Results

Last updated: 2026-06-14

## Current Result

No clean LeWorldModel reset jobs have completed yet. The fixed 24h LR sweep is
now running as Slurm array `3741118_[0-24%12]`. A dependency-held posthoc eval
fallback, `3741137_[0-24%6]`, will run after `3741118` and skip any task whose
integrated diagnostics and planner matrix already exist.

Cancelled/superseded jobs:

- `3740707_[0-24%12]`: not a clean baseline. It trained with 8-frame
  trajectories while planning included horizons up to 64, and its MCTS
  implementation did not perform meaningful tree search at Sudoku root
  branching scale.
- `3741086_[0-24%12]`: first post-fix submission, failed quickly with a BF16
  autocast dtype mismatch in masked projection/sequence encoding. Superseded by
  `3741118`.

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
diagnostics. The BF16 masked projection path is fixed and regression-tested.
Diagnostics now also include train-vs-eval terminal goal-distance checks,
predictor full-vs-truncated BatchNorm deltas, no-history vs full-history latent
rank divergence, branch-prune gold-action survival, latent-rollout drift
against symbolic re-encode by horizon, and planner timing/action-eval counts.

Current submission check at 2026-06-14 14:35 CEST:

- `3741118_0-11` are running on `a40`; `3741118_12-24` are pending due the
  array concurrency cap.
- `3741137_[0-24%6]` is pending on `afterany:3741118` and has its own 24h
  limit for checkpoint-based posthoc eval fallback.
- All active/pending tasks show time limit `1-00:00:00`.
- Task `0` wrote step-1 metrics.
- No tracebacks/errors were found in `logs/lewm_sudoku_lr_3741118_*`; stderr
  contains only the known PyTorch nested-tensor warning.

Verification before this submission:

- `pytest -q` passes.
- `pytest tests/test_lewm_sudoku.py -q` passes with the BF16 autocast regression
  included.
- Tiny train smoke writes scalar metrics plus `diagnostics/` JSONL/CSV/SVG
  artifacts.
- Standalone planner-matrix CLI smoke runs from the smoke checkpoint.
