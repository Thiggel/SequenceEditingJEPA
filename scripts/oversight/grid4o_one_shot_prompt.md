# One-Shot Grid 4O Oversight

You are running as a user-requested one-shot oversight job for
`/home/hpc/c107fa/c107fa12/sequence-editing`. The user explicitly requested
checks at 2h/4h/6h after Grid 4O submission. This does not re-enable recurring
oversight. Do not submit successor oversight jobs.

Follow `AGENTS.md` and the long-form handoff rules. The report source of truth
is `../sequence-editing-report`.

Primary active jobs to check:

- Grid 4O MCTS diagnostics: `3714062_[0-3]`
  - Existing checkpoint:
    `$PUZZLE_JEPA_WORK_ROOT/runs/sudoku_jepa_5m_goal_energy_hwm_l1/checkpoint.pt`
  - Outputs:
    `$PUZZLE_JEPA_WORK_ROOT/runs/sudoku_jepa_5m_goal_energy_hwm_l1/diagnostics_mcts_goal_energy_d8`
    `$PUZZLE_JEPA_WORK_ROOT/runs/sudoku_jepa_5m_goal_energy_hwm_l1/diagnostics_mcts_goal_energy_d16`
    `$PUZZLE_JEPA_WORK_ROOT/runs/sudoku_jepa_5m_goal_energy_hwm_l1/diagnostics_mcts_latent_goal_d8`
    `$PUZZLE_JEPA_WORK_ROOT/runs/sudoku_jepa_5m_goal_energy_hwm_l1/diagnostics_mcts_latent_goal_d16`
- Grid 4M hierarchy value: `3711931_[0-3]`
- Grid 4N macro-action advantage: `3711983`

Checklist:

1. Check `squeue` and `sacct` for the three job groups above.
2. Inspect relevant `logs/puzzle_grid4o_mcts_3714062_*.{out,err}` and any
   Grid 4M/4N logs if those jobs have started.
3. If Grid 4O has partial or completed output, read `diagnostics.json`,
   `mcts_planning_records.jsonl`, and `mcts_debug_records.jsonl`.
4. Analyze whether learned `goal_energy` MCTS improves over beam and whether
   oracle `latent_goal` MCTS works as the search-control. Report solve rate,
   terminal rate, mean remaining Hamming, and root-action debug observations.
5. If there is a clear code bug, fix it surgically, run focused tests, and
   submit a corrected diagnostic job only if needed. Record old/new job IDs.
6. If results are incomplete but logs show progress, record that state without
   over-interpreting.
7. Update handoff docs whenever state, results, interpretation, or submissions
   change:
   - `../sequence-editing-report/STATUS.md`
   - `../sequence-editing-report/BACKLOG.md`
   - `../sequence-editing-report/RESULTS.md`
   - `../sequence-editing-report/LOG.md`
   - `../sequence-editing-report/report.tex`
   - `docs/RUNBOOK.md`
   - `docs/RESULTS.md`
   - `docs/EXPERIMENT_PLAN.md`
8. After successful verification, commit and push both repos. If push fails,
   record the exact failure and leave commits local.

Keep changes minimal and scoped. Do not touch unrelated files or unrelated
projects.
