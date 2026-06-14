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

## 2026-06-14

- 2026-06-14 04:52 CEST oversight read:
  - Requested Grid5B/Grid5C jobs remain out of the live queue. Slurm
    accounting is unchanged: original Grid5B `3724634_[0-5]` hit
    `NODE_FAIL` on `a2143`; rerun `3724689_[0-5]` and original tasks `6-11`
    completed cleanly; all original Grid5C full-matrix evals timed out by wall
    time.
  - Rechecked Grid5C stderrs/stdouts. Stderrs contain only Slurm time-limit
    messages. Stdout epilogues show low continuous GPU/RSS use and
    `/home/vault` above soft quota but below hard quota for those jobs; no
    Python traceback, missing-checkpoint error, hard quota failure, node
    failure, or dependency problem was found.
  - Artifact check is unchanged: all 12 Grid5B run roots have standard,
    MPC-CEM, and symbolic re-encode summaries; no original full-matrix
    `diagnostics_planner_matrix` summaries/records appeared. Small probe
    `3728790` remains the only usable Grid5C matrix artifact with 12 records.
  - Re-aggregated Grid5B/Grid5C reads. Best standard oracle beam remains
    `grid5b_10m_canonical_ema_vicreg_k4`, mean remaining Hamming `29.5625`,
    solve `0/16`; best non-true symbolic re-encode read remains h8 oracle
    `latent_goal`, mean remaining Hamming `41.00`, solve `0/4`. Predicted
    latent MPC-CEM still solves `0`, best proximity `49.5`. Grid5C small probe
    best mode remains MCTS + `symbolic_reencode` + oracle `latent_goal`,
    remaining Hamming `37` from start `55`, solve `0/1`; latent rollout stays
    `53-55`, learned energy `49-54`.
  - Decision unchanged: no Grid5 code change, cancellation, broad Grid5C
    rerun, or hierarchy job is justified. Next Grid5-family work should first
    repair geometry/action ranking or use a tokenized/local control.
  - Oversight job `3724793` completed cleanly in `00:11:18`; `3724794` is the
    active 04:50 CEST check on `a100mig/a0605`; `3724795`-`3724798` remain
    pending by `BeginTime`, so partition broadening cannot help.

## 2026-06-13

- 22:53 CEST oversight read:
  - Requested Grid5B/Grid5C jobs remain out of the live queue. Slurm
    accounting again shows original Grid5B `3724634_[0-5]` as `NODE_FAIL` on
    `a2143`, rerun `3724689_[0-5]` plus original tasks `6-11` as clean
    completions, and all original Grid5C full-matrix tasks as walltime
    timeouts.
  - Rechecked Grid5C stderrs and stdout job statistics. Stderrs contain only
    Slurm time-limit cancellation messages. No Python traceback, quota hard
    failure, missing-checkpoint failure, or dependency problem was found.
    `/home/vault` was over soft quota in the job epilogues but below hard
    quota.
  - Artifact check is unchanged: the original full-matrix jobs left no
    `diagnostics_planner_matrix` summaries/records. The only usable Grid5C
    matrix artifact remains small probe `3728790`.
  - Re-aggregated Grid5B diagnostics. Best standard oracle beam remains
    `grid5b_10m_canonical_ema_vicreg_k4`, mean remaining Hamming `29.5625`,
    solve `0/16`, latent gold-action top1 `0.125`. Best non-true symbolic
    re-encode read remains h8 oracle `latent_goal`, mean remaining Hamming
    `41.00`, solve `0/4`. Predicted-latent MPC-CEM still solves `0`; best
    proximity is `49.5`.
  - Re-read small probe `3728790`: best mode remains MCTS +
    `symbolic_reencode` + oracle `latent_goal`, remaining Hamming `37` from
    start `55`, solve `0/1`. Latent-rollout modes remain `53-55`; learned
    `goal_energy` remains weaker at `49-54`.
  - Decision unchanged: no Grid5 code change, cancellation, broad Grid5C
    rerun, or hierarchy job is justified. The next Grid5-family experiment
    should first repair geometry/action ranking or use a tokenized/local
    control.
  - Oversight job `3724792` completed cleanly in `00:14:45`; `3724793` is the
    active 22:50 CEST check on `a100mig/a0605`; `3724794`-`3724798` remain
    pending by `BeginTime`, so partition broadening would not help.
  - Committed code-repo handoff update as `1331a59` and report-repo handoff
    update as `6d9eccc`. Push failed for both repos with:
    `ssh: connect to host github.com port 22: Connection timed out` and
    `fatal: Could not read from remote repository.`
- 16:50 CEST oversight read:
  - Slurm has no live entries for the requested Grid5B/Grid5C job IDs. Final
    state is unchanged: original Grid5B `3724634_[0-5]` hit `NODE_FAIL` on
    `a2143`; rerun `3724689_[0-5]` and original tasks `6-11` completed
    cleanly; all original Grid5C planner-matrix tasks timed out by wall time.
  - Checked Grid5C stderrs again. Each contains only the Slurm time-limit
    cancellation line; there is still no Python traceback, quota write failure,
    missing-checkpoint failure, or dependency problem.
  - Artifact check found no original full-matrix `diagnostics_planner_matrix`
    summaries/records. The only Grid5 planner-matrix artifact remains the
    small probe `3728790`.
  - Re-read Grid5B/Grid5C summaries. Best Grid5B standard diagnostic remains
    `grid5b_10m_canonical_ema_vicreg_k4`, oracle beam mean remaining Hamming
    `29.5625`, solve `0/16`; best symbolic re-encode non-true score remains
    h8 oracle `latent_goal`, mean remaining Hamming `41.00`, solve `0/4`.
    Predicted-latent MPC-CEM best proximity is still around `49.5`, solve `0`.
  - Decision unchanged: do not rerun or broaden Grid5C, and do not submit
    hierarchy on this compact scorer. The next useful Grid5-family job must
    first repair geometry/action ranking or use a tokenized/local control.
  - Oversight job `3724792` is the active 16:50 CEST check on `a0605`;
    later checks `3724793`-`3724798` remain pending by `BeginTime`.
  - Committed local handoff updates. Push failed for both repos with:
    `ssh: connect to host github.com port 22: Connection timed out` and
    `fatal: Could not read from remote repository.`
- 12:36 CEST status check:
  - Oversight job `3724791` completed cleanly at 2026-06-13 11:08:35 CEST on
    `a0605` (`00:18:05`, exit `0:0`). It did not submit new experiments beyond
    the already-recorded Grid5C streaming probe and geometry probe.
  - Scheduled oversight jobs `3724792`-`3724798` remain pending by `BeginTime`;
    partition broadening cannot help these holds.
  - No new Grid5 artifacts appeared after the 10:56 read. The active decision
    remains: do not broaden Grid5C or add hierarchy on the compact scorer until
    geometry/action ranking is repaired or a tokenized/local control is used.
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
- 10:56 CEST oversight read:
  - Grid5C tasks `3724691_[0-5]` also timed out at 2026-06-13 05:13 CEST.
    Together with `3724698_[9-11]`, `3724700_6`, `3724701_7`, and
    `3724702_8`, the full planner matrix produced no usable summaries. All
    checked stderrs contain only Slurm time-limit messages.
  - Small streaming probe `3728790` completed cleanly in `01:03:08` on a40
    node `a0124` and wrote `planner_summary.json` plus `planner_records.jsonl`.
    On one eval board for `grid5b_10m_canonical_ema_vicreg_k4`, best h8 result
    was MCTS + `symbolic_reencode` + oracle `latent_goal`: start Hamming `55`,
    final remaining Hamming `37`, solve `0/1`. Beam oracle symbolic was `39`;
    learned-energy symbolic was `49` for beam/MCTS and `54` for `nn_cem`;
    latent-rollout modes stayed `53-55`.
  - Added and ran `scripts/analysis/grid5_geometry_probe.py` locally on the
    same checkpoint. Artifact:
    `$PUZZLE_JEPA_WORK_ROOT/analysis/grid5_geometry_probe_canonical_ema_vicreg_k4_20260613/`.
    Read: one-cell terminal corruptions can be extremely close to the true
    terminal latent (`p10` corrupt latent MSE `0.00168`, mean minimum margin
    `0.00047`); learned `goal_energy` ranked the true terminal top-1 in `0/16`
    boards; latent/Hamming nearest-neighbor Spearman was `0.133`; the best
    wrong action displacement had higher goal-direction cosine than the gold
    action in `84.4%` of samples.
  - Interpretation: this follows the Grid5C failure branch. Do not broaden the
    planner matrix or add hierarchy on this compact scorer. The next useful
    experiment should first repair geometry/action ranking or use a
    tokenized/local control.
  - Oversight jobs `3724789` and `3724790` completed cleanly; `3724791` was
    running on `a0605`; later oversight jobs were pending only by `BeginTime`,
    so no partition broadening was applicable.
  - Committed code-repo changes as `afa9918` and report-repo changes as
    `75b3d77`, then attempted `git push` in both repos. Push failed for both
    repos with:
    `ssh: connect to host github.com port 22: Connection timed out` and
    `fatal: Could not read from remote repository.`
