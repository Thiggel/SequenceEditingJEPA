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
- Diagnostics bundle: `puzzle_jepa/eval/lewm_diagnostics.py`
- Planner algorithms: `puzzle_jepa/planning/lewm_planner.py`
- Slurm launcher: `scripts/slurm/run_lewm_sudoku_lr_sweep.slurm`

The live Slurm surface intentionally has one job file. The fixed LR sweep was
submitted as Slurm array `3741086_[0-24%12]` on 2026-06-14 14:24 CEST.
Historical Grid4-Grid6 notes are legacy context only; see
`docs/legacy/README.md` and `../sequence-editing-report/notes/legacy.md`.

## Verify

The LeWM regression tests cover masked BatchNorm padding, full-history latent
rollout, local-search candidate replacement, AdaLN conditioning, goal-independent
state embeddings, latent rollout history-window limits, planner matrix labeling,
planner sanity checks, and diagnostic file generation.

```bash
source scripts/env.sh
pytest -q
python -m py_compile \
  puzzle_jepa/models/lewm.py \
  puzzle_jepa/data/lewm_sudoku.py \
  puzzle_jepa/planning/lewm_planner.py \
  puzzle_jepa/train/lewm_sudoku.py \
  puzzle_jepa/eval/lewm_diagnostics.py \
  puzzle_jepa/eval/lewm_planner_matrix.py
bash -n scripts/slurm/run_lewm_sudoku_lr_sweep.slurm
```

## Submit

The fixed sweep is already submitted:

```bash
squeue -j 3741086
tail -f logs/lewm_sudoku_lr_3741086_0.out
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
`diagnostics.json`, a detailed `diagnostics/` directory, and
`planner_matrix.jsonl`.

Planner-matrix rows include solve/remaining-Hamming metrics plus
`action_evals_*`, `elapsed_seconds_*`, and `seconds_per_action_eval`, so slow
or wasteful planners can be identified directly from `planner_matrix.jsonl`.

Current config trains full fill-only Sudoku trajectories by default
(`training.num_frames: null`) with variable-length masks. `model.max_history`
is `82`, covering full training trajectories. Latent-rollout scoring now caps
the effective lookahead when observed history plus requested horizon would
exceed the predictor context.

Fixed review notes: variable-length masks now keep padded frames out of
encoder/predictor BatchNorm projector statistics; sequence projector BatchNorm
is step-wise so unsupervised future outputs do not change supervised prefix
predictions; state embeddings are encoded from board sequences only and no
longer depend on the `goals` argument; solved frames reuse their state embedding
as the exact goal-distance target; latent-rollout MPC and
score-pruned branch ranking pass observed board/action history into model
rollout; projection panels pass oracle history for latent-rollout scores; local
search updates the same candidate it mutates. `planner="mcts"` is reported as
`score_pruned_progressive_uct` when `mcts_branch_size > 0`.

Additional diagnostics now write train-vs-eval terminal goal-distance checks,
full-vs-truncated predictor BatchNorm deltas, no-history vs full-history latent
rollout rank divergence, branch-prune gold-action survival, latent-rollout vs
symbolic re-encode error by horizon, and planner timing/action-eval counts. The
branch-prune diagnostic enumerates the full immediate action set for up to four
examples by default to keep per-checkpoint eval practical.

The latest red review tests for AdaLN double-normalization, goal-independent
state embeddings, latent MPC history-window handling, and MCTS matrix labels are
green.
