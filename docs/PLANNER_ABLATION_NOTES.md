# Planner Ablation Notes

Last updated: 2026-06-03 09:24 CEST

These notes are deferred. Grid 4B learned-energy reset/beam finished with
`0/128` solves for L1/L2/L3, so the immediate gate is scorer/ranking calibration
or a verifier/goal objective rather than expanding the planner optimizer grid.

## Current Gate

Grid 4B `3691590_[0-2]` tested whether learned goal energy can replace oracle
solved-board latent MSE under the previously successful beam/reset regime:

- beam search over legal Sudoku writes
- symbolic board state
- `--planning-score goal_energy`
- reset/re-encode cadence 4
- L1/L2/L3 Grid 4A checkpoints

Result: all levels solved `0/128`; fix scorer/ranking/calibration before
investing in more CEM or gradient-planning variants.

## Planning Cost

Use energy only for the next planner grid. Do not ablate action costs yet.

For Sudoku, generic continuous action penalties such as `||a||^2` are not
natural because primitive actions are categorical `(row, col, value)` writes.
Overwrite/edit/conflict/blank penalties may become useful later, but they would
confound the next comparison. The next grid should isolate the optimizer and
outer-loop choice under the same learned-energy objective.

## Deferred Optimizer Grid

Use MPC-style replanning rather than full-horizon one-shot planning.

Candidate outer-loop settings:

- `H=8, k=1`: plan 8 edits, execute 1, re-encode/replan.
- `H=8, k=4`: plan 8 edits, execute 4, re-encode/replan.
- `H=16, k=4`: plan 16 edits, execute 4, re-encode/replan.

Here `H` is the planning horizon and `k` is the executed prefix length.
Mutable-cell overwrites should remain allowed; clue overwrites should remain
forbidden.

Inner optimizers to compare:

- Beam search.
- CEM over primitive Sudoku edit chunks.
- Gradient descent over high-level continuous latent actions, then CEM for
  low-level primitive actions to reach the induced subgoal.
- Gradient descent over all action variables, using a soft/continuous low-level
  action relaxation followed by nearest-neighbor decoding to discrete
  `(row, col, value)` actions.

Keep the objective energy-only across these variants.

## Complexity Reminder

For MPC-CEM, with total executed edit budget `T`, horizon `H`, population `P`,
iterations `I`, and executed prefix `k`, approximate sampled action-step cost is:

```text
O((T / k) * I * P * H)
```

Example: `T=128`, `H=8`, `P=256`, `I=4`, `k=4` is about `262k` sampled
action steps before model batching overhead.
