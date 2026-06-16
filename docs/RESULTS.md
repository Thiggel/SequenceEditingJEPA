# Results

Last updated: 2026-06-16 13:10 CEST

## Current Result

The fixed LeWorldModel LR sweep has produced usable checkpoints for LR indices
`0-23`. LR `7e-4` / task `3741118_24` wrote checkpoints but became numerically
invalid (`NaN` losses from about step `13000`) and failed diagnostic PCA/SVD, so
exclude it from planner comparison.

Current split planner eval state at 2026-06-16 13:10 CEST:

- completed: exact symbolic (`3745945`), categorical CEM (`3745942`), local
  search (`3745943`).
- running/partial: beam (`3745940`), best-first (`3745941`), MCTS /
  score-pruned progressive UCT (`3745944`).

Current aggregate read:

- exact symbolic baseline solves all boards.
- beam, best-first, greedy, and MCTS solve all boards when scored by exact
  `true_hamming_oracle`, which confirms the fill-only mechanics are sound.
- `oracle_goal_distance` and `predicted_goal_distance` still solve `0/128`
  across written beam/best-first/MCTS/CEM/local-search rows and typically end
  near `48` remaining cells.
- categorical CEM and local search are not strong Sudoku planners under the
  current settings: even true-Hamming oracle scoring leaves about `40.35` and
  `39.35` cells wrong, respectively.
- MCTS now has enough written rows to judge the main signal: true-Hamming
  oracle solves, but latent oracle distance and learned predicted distance
  remain `0.0` solve rate with about `48` remaining cells.

Interpretation: the stronger discrete planners do not rescue the latent or
learned value score. The exact score can drive beam/best-first/MCTS to a
solution, but Euclidean solved-board latent distance and the learned scalar
distance are still not reliable action-selection objectives for Sudoku.

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
