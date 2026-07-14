# Puzzle JEPA

This repo is focused on JEPA-style latent world models for objective puzzle and
grid-edit reasoning. The current experiment uses valid rigid-object motion,
one MLP-compressed state latent, and a fixed `[1,10,100]` latent-world-model
hierarchy. Only the EMA+VICReg coefficient sweep is active.

Long-form status, backlog, results, and the ongoing LaTeX report live in
`../sequence-editing-report`. In-repo operational state is in
`docs/CURRENT_EXPERIMENTS.md`, the staged backlog is `docs/BACKLOG.md`, and all
historical waves are indexed by `docs/experiments/README.md`.

## Current Controlled HWM

Core files:

- `puzzle_jepa.controlled_objects`: valid rigid-object domain, trajectories,
  MLP JEPA, hierarchy, probes, and planners.
- `puzzle_jepa.train.controlled_objects`: staged Hydra trainer.
- `configs/controlled_objects`: fixed valid-motion model/objective configs.
- `scripts/experiments/submit_controlled_objects_vicreg_hwm.sh`: dry-run by
  default; the active arrays are already submitted and must not be duplicated.

See `docs/experiments/WAVE-14-VALID-HWM-VICREG.md` for the exact contract.

## Historical Object Dynamics Branch

Purpose: test whether a compressed single-CLS LeWM/JEPA model trained on
low-level grid edit dynamics can recover object/process abstractions without
object slots or proposal IDs.

Core files:

- `puzzle_jepa.object_dynamics`: hidden-object generator, low-level action
  domain, batching, single-CLS JEPA model, regularizers, and frozen probes.
- `puzzle_jepa.train.object_dynamics`: Hydra trainer for rollout, hierarchy,
  LDAD, EMA, VICReg, and SIGReg variants.
- `configs/object_dynamics`: Hydra groups for `data`, `model`, `objective`,
  and sweep metadata.
- `scripts/slurm/run_object_dynamics_train.slurm`: one training job template.
- `scripts/experiments/submit_object_dynamics_prestage.sh`: dry-run LR/step
  prestage grid.
- `scripts/experiments/submit_object_dynamics_phase1.sh`: dry-run trajectory
  and model/objective sweep.

Smoke run:

```bash
source scripts/env.sh
PUZZLE_JEPA_WORK_ROOT=/tmp/puzzle_jepa_object_dynamics_smoke \
python -m puzzle_jepa.train.object_dynamics \
  --config-name train \
  data=object_blocked model=cls64_r1 objective=base \
  output_dir=/tmp/puzzle_jepa_object_dynamics_smoke/run \
  training.max_steps=1 training.batch_size=2 \
  eval.probe_train_samples=8 eval.probe_eval_samples=6 eval.probe_steps=2
```

These historical job grids remain reproducible but are superseded:

```bash
scripts/experiments/submit_object_dynamics_prestage.sh
scripts/experiments/submit_object_dynamics_phase1.sh
```

Do not submit them while the controlled HWM sweep is active.

## Legacy Puzzle Surfaces

The previous sequence-editing/iGSM work was archived outside the repo at
`../legacy-sequence-editing` and summarized in
`../sequence-editing-report/notes/legacy.md`. Older Sudoku and ARC scaffolds are
kept because they are referenced by tests, results, and the report.

The older scaffold has these pieces:

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

First-pass ARC train jobs:

```bash
scripts/experiments/submit_arc_jepa_firstpass.sh
```

Render ARC proposal/action example diagrams:

```bash
python scripts/analysis/render_arc_example_diagrams.py
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
