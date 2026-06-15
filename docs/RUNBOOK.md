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
- Slurm training launcher: `scripts/slurm/run_lewm_sudoku_lr_sweep.slurm`
- Slurm posthoc eval fallback:
  `scripts/slurm/run_lewm_sudoku_posthoc_eval.slurm`

The live Slurm surface has one training job file plus one dependency-held
posthoc eval fallback. The fixed LR sweep is running as Slurm array
`3741118_[0-24%12]`, submitted on 2026-06-14 14:33 CEST with a 24h time limit.
The eval fallback is `3741137_[0-24%6]`, submitted with
`--dependency=afterany:3741118` and its own 24h time limit.
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
bash -n scripts/slurm/run_lewm_sudoku_posthoc_eval.slurm
```

## Submit

The fixed sweep is already submitted:

```bash
squeue -j 3741118
squeue -j 3741137
tail -f logs/lewm_sudoku_lr_3741118_0.out
```

The dependency-held eval fallback `3741137_[0-24%6]` runs after the whole
training array finishes. Each task checks the matching run root; if
`diagnostics.json` and `planner_matrix.jsonl` already exist, it exits without
doing work. If integrated eval timed out after `checkpoint.pt` was written, it
runs the diagnostics bundle plus fast planner-matrix eval into `posthoc_eval/`.

Cancelled/superseded submissions:

- `3740707_[0-24%12]`: trained with 8-frame subtrajectories and pre-fix MCTS,
  so it should not be used as the clean LeWM result. It was cancelled on
  2026-06-14 before code fixes; tasks `0-11` ran for about 21 minutes and tasks
  `12-24` never started.
- `3741086_[0-24%12]`: first post-fix submission. Tasks `0-23` failed quickly
  with a BF16 autocast dtype mismatch in masked projection/sequence encoding;
  task `24` was cancelled before running. Superseded by `3741118`.

The current array sweeps 25 learning rates:

```text
1e-6..9e-6, 1e-5..9e-5, 1e-4..7e-4
```

Run roots are written under:

```text
$PUZZLE_JEPA_WORK_ROOT/runs/lewm_sudoku_lr_<lr>
```

Current submission health at 2026-06-15 16:18 CEST: tasks `0-11` timed out at
24h during integrated planner eval after writing checkpoints, diagnostics, and
partial `planner_matrix.jsonl` files. Tasks `12-23` are running on `rtxpro6k`
at about 1.75h elapsed and have already reached `step=20000`; task `24` is
pending due `JobArrayTaskLimit`. Eval fallbacks `3741137_[0-24%6]` and
corrected fallback `3742630_[0-24%6]` are pending on dependency. No
tracebacks/errors were found in `logs/lewm_sudoku_lr_3741118_*`; stderr
contains the known PyTorch nested-tensor warning plus expected Slurm timeout
messages for tasks `0-11`.

Interim result: LRs `2e-5` and `3e-5` currently have the best step-20k value
metrics, with value RMSE about `0.39`/`0.34` and value correlation about
`0.991`. The planner matrix rows written so far solve the 4-example fast set
with `true_hamming_oracle`, but `oracle_goal_distance` and
`predicted_goal_distance` are still `0/4` in the greedy/beam rows already
written, even with symbolic re-encode. The slow tail is latent-distance
planning: for LR `2e-5`, beam + symbolic re-encode + oracle latent distance
at horizon 64 took about 8,121s over four examples, and latent-rollout beam
horizon 8 took about 6,007s.
The first-wave integrated matrices reached only greedy plus part of beam; no
CEM, local-search, MCTS, best-first, or exact rows were written before timeout.

Fallback note: the original fallback `3741137` can incorrectly skip a partial
planner matrix because it checks only nonempty `diagnostics.json` and
`planner_matrix.jsonl`. Corrected fallback `3742630` uses top-level
`metrics.json` as the completion sentinel and will produce `posthoc_eval/`
for any run whose integrated trainer eval timed out after checkpointing.

Split planner-only eval arrays were submitted on 2026-06-15 with
`--dependency=afterany:3741118`, one 24h array per planner and 25 LR tasks per
array. These use `--skip-diagnostics` and write to
`$PUZZLE_JEPA_WORK_ROOT/runs/lewm_sudoku_lr_<lr>/posthoc_planners/<planner>/`.
They are intended to get CEM/MCTS/best-first/local-search rows without letting
beam block the whole matrix:

- `3745791_[0-24%25]`: greedy
- `3745792_[0-24%25]`: beam
- `3745793_[0-24%25]`: best-first
- `3745794_[0-24%25]`: categorical CEM
- `3745795_[0-24%25]`: local search
- `3745796_[0-24%25]`: MCTS / score-pruned progressive UCT
- `3745797_[0-24%25]`: exact symbolic

The older broad fallbacks `3741137` and `3742630` are still pending; the split
planner arrays are the preferred outputs for planner comparisons.

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
green. A BF16 masked projection regression is also covered by
`test_masked_forward_supports_bfloat16_autocast`.
