# Results

Last updated: 2026-06-15

## Current Result

The first wave of the fixed LeWorldModel LR sweep has reached the final
training step (`20000`) for tasks `0-11` and has written checkpoints,
`diagnostics.json`, and partial fast planner matrices for LRs `1e-6` through
`3e-5`. The tasks are still running because the integrated planner matrix is
very slow in latent-distance beam rows. Tasks `12-24` remain pending behind
`JobArrayTaskLimit`; dependency fallback `3741137_[0-24%6]` remains pending on
`afterany:3741118`.

Current health at 2026-06-15 16:18 CEST:

- `3741118_0-11`: timed out at 24h during integrated planner eval. They wrote
  checkpoints, diagnostics, and partial planner matrices but no top-level
  `metrics.json`.
- `3741118_12-23`: running on `rtxpro6k`, elapsed about 1.75h. These tasks
  have already reached `step=20000` and are likely in diagnostics/planner eval.
- `3741118_24`: pending due array concurrency cap.
- `3741137_[0-24%6]`: older fallback pending due dependency; its guard only
  checks nonempty diagnostics/planner files, so it may skip partial matrices.
- `3742630_[0-24%6]`: corrected fallback pending due dependency; it skips only
  when top-level `metrics.json` exists, which means the integrated trainer eval
  returned.
- No tracebacks/errors in `logs/lewm_sudoku_lr_3741118_*`; stderr only shows
  the known PyTorch nested-tensor warning plus expected Slurm timeout messages
  for tasks `0-11`.

Preliminary first-wave read:

- Best training/value metrics among completed step-20k records are currently
  around LR `2e-5`/`3e-5`: value RMSE about `0.39`/`0.34`, value correlation
  about `0.991`, prediction loss about `0.042`/`0.088`.
- Latent geometry looks healthier than the lowest LR collapse case: effective
  rank is roughly `11-14` for `9e-6` to `3e-5`, versus about `1.2` for `1e-6`.
- Planner rows written so far solve the 4-example fast matrix only with
  `true_hamming_oracle`. Both `oracle_goal_distance` and
  `predicted_goal_distance` are `0/4` in the written greedy/beam rows, even
  with symbolic re-encode.
- Runtime is the immediate issue: for LR `2e-5`, beam + symbolic re-encode +
  oracle latent goal distance took about 8,121s for horizon 64 over only four
  examples; latent-rollout beam horizon 8 already took about 6,007s.
- The first-wave integrated matrices reached only greedy plus part of beam:
  no categorical CEM, local search, MCTS, best-first, or exact rows were written
  before timeout.

Planner-only eval resubmission:

- `3745791_[0-24%25]`: greedy
- `3745792_[0-24%25]`: beam
- `3745793_[0-24%25]`: best-first
- `3745794_[0-24%25]`: categorical CEM
- `3745795_[0-24%25]`: local search
- `3745796_[0-24%25]`: MCTS / score-pruned progressive UCT
- `3745797_[0-24%25]`: exact symbolic

All are held on `afterany:3741118`, have 24h limits, skip diagnostics, and
write planner rows under each LR run root in `posthoc_planners/<planner>/`.

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
