# Results

Last updated: 2026-06-16 15:05 CEST

## Current Result

No Grid-Token Goal-JEPA jobs have been submitted yet.

Implementation and verification are complete:

- Active Slurm jobs were cancelled before the refactor.
- Old LeWM/CLS/value-head code, tests, configs, and Slurm launchers were
  removed from the active surface.
- New Grid-Token Goal-JEPA model/data/train/eval/planner path is implemented.
- Active tests pass: `pytest -q` -> `8 passed`.
- Python compilation/import checks pass.

## Legacy Result

The previous faithful LeWM/CLS/value-head reset is now legacy. Its main result
was negative for Sudoku planning geometry: exact symbolic and true-Hamming
oracle scoring could solve, but oracle latent distance and learned scalar
goal-distance scoring did not produce solves. That result motivated the current
full-grid goal-prediction architecture.
