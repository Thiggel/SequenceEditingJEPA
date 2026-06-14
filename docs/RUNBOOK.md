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

The live Slurm surface intentionally has one job file. Do not submit it until
the four red LeWM review tests pass and the user says `go`.
Historical Grid4-Grid6 notes are legacy context only; see
`docs/legacy/README.md` and `../sequence-editing-report/notes/legacy.md`.

## Verify

The LeWM regression tests cover masked BatchNorm padding, full-history latent
rollout, local-search candidate replacement, planner sanity checks, and
diagnostic file generation. Four newest review tests are intentionally red until
the remaining blockers are fixed:

- `test_adaln_modulation_is_not_renormalized_inside_sublayers`
- `test_forward_state_embeddings_do_not_depend_on_goal_argument`
- `test_latent_rollout_mpc_replanning_respects_predictor_history_limit`
- `test_planner_matrix_records_mcts_variant_name`

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

Do not submit jobs until those four tests are fixed and the user explicitly says
`go`.

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
`diagnostics.json`, a detailed `diagnostics/` directory, and
`planner_matrix.jsonl`.

Planner-matrix rows include solve/remaining-Hamming metrics plus
`action_evals_*`, `elapsed_seconds_*`, and `seconds_per_action_eval`, so slow
or wasteful planners can be identified directly from `planner_matrix.jsonl`.

Current config trains full fill-only Sudoku trajectories by default
(`training.num_frames: null`) with variable-length masks. `model.max_history`
is `82`, covering full training trajectories, but latent-rollout MPC still needs
a fix for replans where observed history plus the requested search horizon
exceeds this window.

Fixed review notes: variable-length masks now keep padded frames out of
encoder/predictor BatchNorm projector statistics; sequence projector BatchNorm
is step-wise so unsupervised future outputs do not change supervised prefix
predictions; goal boards are encoded in the same training-mode BatchNorm pass as
trajectory states for goal-distance targets; latent-rollout MPC and
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

Open blockers: AdaLN has extra post-modulation LayerNorms, training embeddings
depend on the `goals` argument through shared BatchNorm context, latent MPC can
overrun `max_history`, and planner-matrix rows do not use the MCTS variant label.
