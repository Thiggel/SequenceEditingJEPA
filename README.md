# Puzzle JEPA

This repo is now focused on objective puzzle-world reasoning for Sudoku,
Maze-Hard, and ARC-style candidate-output refinement. The previous
sequence-editing/iGSM work was archived outside the repo at
`../legacy-sequence-editing` and summarized in
`../sequence-editing-report/notes/legacy.md`.

Long-form status, backlog, results, and the ongoing LaTeX report live in
`../sequence-editing-report`. The `docs/` files in this repo are compact
operational pointers only.

The active scaffold has four pieces:

- `puzzle_jepa.data`: Sudoku and maze state/action worlds, ARC grid/task
  loaders, ARC proposal/action scaffolding, Hugging Face string adapters,
  oracle transition sampling, and tensor collation.
- `puzzle_jepa.models`: minimal HRM, TRM, PTRM sampler, and a decoder-free
  action-conditioned JEPA world model.
- `puzzle_jepa.planning`: symbolic action enumeration plus latent action scoring
  against an oracle goal state.
- `puzzle_jepa.eval.arc_oracle_coverage`: CPU-only ARC proposal/action coverage
  analyzer used before any ARC model training.
- `configs/puzzle`: Hydra smoke configs for JEPA, HRM, TRM, and PTRM.

## Setup

```bash
source scripts/env.sh
python -m pytest -q tests
```

## ARC Coverage Probe

The first ARC implementation is intentionally non-neural. It checks whether the
state/action interface is concrete enough before training a value model or
JEPA:

```bash
python scripts/analysis/arc_oracle_coverage.py \
  --data-root /path/to/arc-agi \
  --split training \
  --limit 50 \
  --max-episodes-per-task 2 \
  --max-depth 1 \
  --beam-width 4 \
  --no-cell-actions
```

## Smoke Runs

```bash
python -m puzzle_jepa.train.hydra_train --config-name jepa_sudoku_smoke
python -m puzzle_jepa.train.hydra_train --config-name jepa_maze_smoke
python -m puzzle_jepa.train.hydra_train --config-name hrm_sudoku_smoke
python -m puzzle_jepa.train.hydra_train --config-name trm_sudoku_smoke
python -m puzzle_jepa.train.hydra_train --config-name ptrm_sudoku_smoke
```

## Current Training Direction

Start with valid oracle partial states:

1. Sample a puzzle and its oracle solution/path.
2. Sample a valid partial state from the solution manifold.
3. Enumerate a legal action `(row, col, value)`.
4. Train the JEPA predictor to map `(state, action)` to the target-encoder latent
   of the next state.
5. Plan by scoring legal actions by predicted latent distance to the oracle goal
   latent.

Invalid states should be added later as verifier/value-head negatives, not mixed
silently into the world-model transition loss.
# sequence-editing-report
