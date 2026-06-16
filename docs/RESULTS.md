# Results

Last updated: 2026-06-16 16:44 CEST

## Current Result

No Grid-Token Goal-JEPA jobs have been submitted yet.

Implementation and review fixes are complete:

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
- Current test command:
  `source scripts/env.sh && pytest -q` -> `18 passed`.
- Running `pytest -q` without `source scripts/env.sh` fails at collection
  because the default Python cannot import `torch`.

## Legacy Result

The previous faithful LeWM/CLS/value-head reset is now legacy. Its main result
was negative for Sudoku planning geometry: exact symbolic and true-Hamming
oracle scoring could solve, but oracle latent distance and learned scalar
goal-distance scoring did not produce solves. That result motivated the current
full-grid goal-prediction architecture.
