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
- 22:52 CEST oversight read:
  - Grid5B rerun `3724689_[0-5]` completed cleanly; all final Grid5B tasks
    are now complete. Stderrs checked for final runs are empty.
  - Grid5B did not pass the solve gate. Best symbolic re-encode oracle
    proximity is `grid5b_10m_canonical_ema_vicreg_k4`, h8 mean remaining
    Hamming `41.00`, root goal-value rate `0.500`, solve `0/4`; its cheap
    beam diagnostic has oracle mean remaining Hamming `29.56` and latent
    top-goal-value rate `0.969`, but exact solves remain `0`.
  - Predicted-latent MPC-CEM still solved `0` for every Grid5B variant; best
    proximity is `grid5b_10m_canonical_ema_sigreg_k4`, h64 `goal_energy`,
    mean remaining Hamming `49.50`.
  - Perfect true-Hamming symbolic CEM gets near the solution for several
    Grid5B runs, including mean remaining Hamming `1.75` and solve `1/4` for
    `canonical_ema_vicreg_k4`, `oldbest_scaled_ema_sigreg_k4`, and
    `oldbest_scaled_sigreg_k4`. This keeps planner/action factorization on
    the table, but latent and learned scores remain the blocker.
  - All Grid5C planner jobs are running. Stderrs are empty and the tasks show
    CPU/RSS activity, but no `diagnostics_planner_matrix/planner_summary.json`
    files exist yet. No new experiment was submitted while this gate is
    pending.
  - Committed code-repo docs as `552657b` and report-repo docs as `c525344`.
    Push failed for both repos with:
    `ssh: connect to host github.com port 22: Connection timed out` and
    `fatal: Could not read from remote repository.`

## 2026-06-13

- 04:54 CEST oversight read:
  - Grid5C tasks `3724698_[9-11]`, `3724700_6`, `3724701_7`, and
    `3724702_8` timed out before writing `planner_summary.json`. Their
    stderrs contain only Slurm time-limit messages; stdout job statistics show
    low but continuous GPU/RSS use and no Python traceback. `/home/vault`
    remains over soft quota but below hard quota, so this was runtime loss, not
    a quota write failure.
  - Grid5C tasks `3724691_[0-5]` were still running on `a40` at elapsed
    `11:41/12:00`. `scontrol update JobId=3724691 TimeLimit=24:00:00` failed
    with `Access/permission denied`, so no walltime rescue was possible.
  - Fixed `puzzle_jepa/eval/grid5_planner_matrix.py` to write
    `planner_records.jsonl` and `planner_summary.json` after every completed
    mode instead of only at process end. Added a focused test and
    `scripts/slurm/run_grid5c_planner_matrix_probe.slurm`.
  - Verification passed: `source scripts/env.sh && pytest
    tests/test_grid5_sigreg.py -q`, `python -m py_compile
    puzzle_jepa/eval/grid5_planner_matrix.py`, `bash -n` for both Grid5C Slurm
    wrappers, and a real-checkpoint CLI smoke under
    `$PUZZLE_JEPA_WORK_ROOT/analysis/grid5_planner_matrix_incremental_smoke_20260613/`.
  - Submitted smallest streaming diagnostic `3728790`, targeting
    `grid5b_10m_canonical_ema_vicreg_k4` with one board, h8, all optimizer /
    transition / score axes, reduced budgets, and `a2143` excluded. It was
    running on `a40` node `a0124` at the handoff.
- 05:03 CEST push attempt:
  - Committed code-repo changes as `e5c9a37` and report-repo changes as
    `9763418`, then attempted `git push` in both repos.
  - Push failed for both repos with:
    `ssh: connect to host github.com port 22: Connection timed out` and
    `fatal: Could not read from remote repository.`
