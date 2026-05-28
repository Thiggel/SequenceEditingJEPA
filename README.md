# Puzzle JEPA

This repo is now focused on objective puzzle-world reasoning for Sudoku-Extreme
and Maze-Hard. The previous sequence-editing/iGSM work was archived outside the
repo at `../legacy-sequence-editing` and summarized in [`legacy.md`](legacy.md).

The active scaffold has four pieces:

- `puzzle_jepa.data`: Sudoku and maze state/action worlds, Hugging Face string
  adapters, oracle transition sampling, and tensor collation.
- `puzzle_jepa.models`: minimal HRM, TRM, PTRM sampler, and a decoder-free
  action-conditioned JEPA world model.
- `puzzle_jepa.planning`: symbolic action enumeration plus latent action scoring
  against an oracle goal state.
- `configs/puzzle`: Hydra smoke configs for JEPA, HRM, TRM, and PTRM.

## Setup

```bash
source scripts/env.sh
python -m pytest -q tests
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
