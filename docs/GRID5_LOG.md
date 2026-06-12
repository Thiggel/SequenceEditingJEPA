# Grid 5 Log

Concise Grid-5-era log. Full historical logs remain in
`../sequence-editing-report/LOG.md`.

## Lessons So Far

- Tokenized/local older models could solve Sudoku under oracle solved-board
  latent scoring when candidate symbolic boards were periodically re-encoded.
- Learned terminal-energy/value heads repeatedly failed to replace the oracle
  solved-board latent metric.
- Compact single-state Grid 5 models with JEPA latent MSE plus SIGReg did not
  solve under small beam planning, MPC-CEM, recursive rollout training, or local
  symbolic re-encode probes.
- SIGReg/VICReg-style stabilization is necessary to avoid collapse but is not
  by itself a planning metric.
- The current open question is whether 10M capacity/stabilization plus better
  planner structure makes the compact latent usable, or whether the geometry is
  fundamentally misaligned with symbolic constraint planning.

## 2026-06-12

- Submitted Grid 5B 10M stabilizer/capacity screen.
- Original Grid 5B tasks `0-5` hit Slurm `NODE_FAIL` on `a2143`; rerun
  submitted as `3724689_[0-5]` excluding that node.
- Implemented and submitted Grid 5C planner matrix:
  - `beam`, `mcts`, `nn_cem`;
  - `symbolic_reencode` vs `latent_rollout`;
  - oracle `latent_goal` vs learned `goal_energy`.
- Added this clean Grid5 plan/backlog/log layer so future decisions are gated
  by current Grid5 results rather than the full pre-Grid5 history.
- Tested oversight invocation:
  - `3724771` failed because `codex cs` is not a real CLI subcommand.
  - `3724773` failed because approval/sandbox flags were placed after
    `codex exec` for this CLI version.
  - `3724777` passed using `codex exec` directly with medium reasoning.
  - `3724784` failed because aliases do not expand behind `timeout`.
  - `3724787` passed using the local `cs` alias from `~/.bash_profile` as
    `cs ... exec` with medium reasoning.
- Scheduled Grid5 oversight checks every 6h for 2.5 days:
  `3724789`, `3724790`, `3724791`, `3724792`, `3724793`, `3724794`,
  `3724795`, `3724796`, `3724797`, `3724798`.
- Original Grid5B tasks `3724634_6` and `3724634_7` completed cleanly; Grid5C
  planner eval tasks `3724700_6` and `3724701_7` started. Grid5C tasks
  `6-11` are now running; eval `0-5` waits on rerun `3724689`.
