# Results

Last updated: 2026-06-16 17:16 CEST

## Current Result

No Grid-Token Goal-JEPA jobs have been submitted yet.

Implementation review status:

- Active Slurm jobs were cancelled before the refactor.
- New Grid-Token Goal-JEPA model/data/train/eval/planner path is implemented.
- Action-rank positives are now sampled explicitly as target-consistent
  solution fills, independent of random dynamics trajectories.
- `R1_no_context_masks` zeros context values as well as masks, and
  `encode_context` is value-blind when masks indicate no-context mode.
- Model `forward` derives row/column/token counts from inputs instead of
  hard-coding `9x9/81`.
- Remaining legacy CLS/value/causal modules and old grid train/eval/analysis
  paths were removed from the active tree.
- Progress ranking now receives `oracle_mask`; by default it applies to no
  rows, and training passes the true successful-trajectory mask.
- Action ranking now compares distances of encoded symbolic successor boards
  `f_theta(T(s,a),H_c)`, not predictor rollout latents.
- Diagnostics now include predictor rollout drift by horizon, latent-rollout
  top-positive action accuracy, predicted-goal vs oracle-goal alignment,
  distance-vs-Hamming Spearman correlation, action margins by fill depth, and
  terminal corruption margins by corruption size.
- HRM/TRM scaffolding remains intentionally as future baselines.
- Action-rank training now samples rank states from valid trajectory frames,
  not only the initial puzzle state.
- Current test command:
  `source scripts/env.sh && pytest -q` -> `26 passed`.
- Running `pytest -q` without `source scripts/env.sh` fails at collection
  because the default Python cannot import `torch`.

Final-review objective issues are fixed. No Grid-Token jobs have been
submitted.

Planner runtime risk remains: the largest beam matrix settings expand many
unbatched successor scores and may exceed the 24h eval limit.

## Legacy Result

The previous faithful LeWM/CLS/value-head reset is now legacy. Its main result
was negative for Sudoku planning geometry: exact symbolic and true-Hamming
oracle scoring could solve, but oracle latent distance and learned scalar
goal-distance scoring did not produce solves. That result motivated the current
full-grid goal-prediction architecture.
