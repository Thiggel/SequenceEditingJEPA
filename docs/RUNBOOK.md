# Runbook

Last updated: 2026-06-14

Long-form handoff source of truth: `../sequence-editing-report`.

## Active Surface

This repo has been reset to one active experiment path: a LeWorldModel-faithful
Sudoku JEPA.

- Config: `configs/puzzle/lewm_sudoku.yaml`
- Model: `puzzle_jepa/models/lewm.py`
- Data sampler: `puzzle_jepa/data/lewm_sudoku.py`
- Trainer: `puzzle_jepa/train/lewm_sudoku.py`
- Planner/eval matrix: `puzzle_jepa/eval/lewm_planner_matrix.py`
- Planner algorithms: `puzzle_jepa/planning/lewm_planner.py`
- Slurm launcher: `scripts/slurm/run_lewm_sudoku_lr_sweep.slurm`

The live Slurm surface intentionally has one job file. Do not submit it until
the masked BatchNorm and local-search review blockers in
`../sequence-editing-report/BACKLOG.md` are fixed and the user says `go`.
Historical Grid4-Grid6 notes are legacy context only; see
`docs/legacy/README.md` and `../sequence-editing-report/notes/legacy.md`.

## Verify

```bash
source scripts/env.sh
pytest -q
python -m py_compile \
  puzzle_jepa/models/lewm.py \
  puzzle_jepa/data/lewm_sudoku.py \
  puzzle_jepa/planning/lewm_planner.py \
  puzzle_jepa/train/lewm_sudoku.py \
  puzzle_jepa/eval/lewm_planner_matrix.py
bash -n scripts/slurm/run_lewm_sudoku_lr_sweep.slurm
```

## Submit

Do not submit jobs until the review blockers are fixed and the user explicitly
says `go`.

```bash
sbatch scripts/slurm/run_lewm_sudoku_lr_sweep.slurm
```

Cancelled/superseded submission: `3740707_[0-24%12]`. It trained with
8-frame subtrajectories and pre-fix MCTS, so it should not be used as the clean
LeWM result. It was cancelled on 2026-06-14 before code fixes; tasks `0-11`
ran for about 21 minutes and tasks `12-24` never started.

The array sweeps 25 learning rates:

```text
1e-6..9e-6, 1e-5..9e-5, 1e-4..7e-4
```

Run roots are written under:

```text
$PUZZLE_JEPA_WORK_ROOT/runs/lewm_sudoku_lr_<lr>
```

Each run writes `config.json`, `metrics.jsonl`, `checkpoint.pt`,
`diagnostics.json`, and `planner_matrix.jsonl`.

Current config trains full fill-only Sudoku trajectories by default
(`training.num_frames: null`) with variable-length masks. `model.max_history`
is `82`, so planning horizons up to 64 are no longer beyond the trained
positional range for the loaded Sudoku boards.

Review note: those variable-length masks currently do not prevent padded frames
from entering BatchNorm projector statistics, so this needs a code fix before a
clean LR sweep.
