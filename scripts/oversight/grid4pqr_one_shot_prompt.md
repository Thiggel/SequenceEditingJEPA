# One-Shot Grid 4P/4Q/4R Oversight

You are running as a user-requested one-shot oversight job for
`/home/hpc/c107fa/c107fa12/sequence-editing`. The user explicitly requested
checks at these absolute Europe/Berlin times: 2026-06-10 18:00,
2026-06-10 20:00, 2026-06-11 00:00, 2026-06-11 04:00, and
2026-06-11 08:00. This does not re-enable recurring oversight. Do not submit
successor oversight jobs.

Follow `AGENTS.md` and the long-form handoff rules. The report source of truth
is `../sequence-editing-report`.

Primary active jobs to check:

- Grid 4M hierarchy value training/eval: `3711931_[0-3]`
- Grid 4N macro-action advantage training/eval: `3711983`
- Grid 4P streaming MCTS diagnostics: `3715249_[0-3]` on `a40,rtxpro6k`
- Grid 4Q recursive hierarchy diagnostics on Grid 4M checkpoints:
  `3715252_[0-11]`, dependency `afterok:3711931`
- Grid 4R recursive hierarchy diagnostics on Grid 4N checkpoint:
  `3715251_[0-2]`, dependency `afterok:3711983`
- New absolute-time one-shot oversight jobs submitted on 2026-06-10:
  see `docs/RUNBOOK.md` and `../sequence-editing-report/STATUS.md` for ids.

Checklist:

1. Check `squeue` and `sacct` for Grid 4M/4N/4P/4Q/4R and the oversight job
   itself.
   Also confirm the job inherited proxy variables before any network-dependent
   work: the Slurm wrapper sources `scripts/env.sh`, which exports
   `http_proxy`, `https_proxy`, `HTTP_PROXY`, and `HTTPS_PROXY`. If Codex/API
   connectivity itself is unavailable, record that as a network failure. If
   analysis succeeds but `git push` fails with `ssh: connect to host github.com
   port 22: Connection timed out`, treat that as the known GitHub SSH egress
   problem: record the exact failure, leave commits local, and do not submit a
   replacement oversight job solely for that push failure.
2. Inspect relevant logs under `logs/`, especially
   `puzzle_grid4p_mcts_*`, `puzzle_grid4q_recur_*`, and
   `puzzle_grid4r_recur_*`.
3. For Grid 4P, inspect streamed `mcts_planning_records.jsonl` and
   `mcts_debug_records.jsonl` as soon as they exist, even if the Slurm task is
   still running or later times out.
4. For Grid 4Q/4R, inspect `diagnostics.json` and
   `recursive_hierarchical_subgoal_records.jsonl` when available. Compare
   `cem`, `gd`, and `gd_reachability`; record solve rate, terminal rate,
   remaining Hamming, and whether the learned top score appears directionally
   useful.
5. If there is a clear code bug, fix it surgically, run focused tests, and
   submit a corrected diagnostic job only if needed. Record old/new job IDs.
6. If jobs are pending due GPU maintenance, check `sinfo`. Broaden pending
   diagnostic jobs to freer suitable partitions when this is safe. Do not alter
   dependency- or begin-time-blocked oversight jobs where broadening cannot help.
7. If results are incomplete but logs show progress, record that state without
   over-interpreting.
8. Update handoff docs whenever state, results, interpretation, or submissions
   change:
   - `../sequence-editing-report/STATUS.md`
   - `../sequence-editing-report/BACKLOG.md`
   - `../sequence-editing-report/RESULTS.md`
   - `../sequence-editing-report/LOG.md`
   - `../sequence-editing-report/report.tex`
   - `docs/RUNBOOK.md`
   - `docs/RESULTS.md`
   - `docs/EXPERIMENT_PLAN.md`
9. After successful verification, commit and push both repos. If push fails,
   record the exact failure and leave commits local.

Keep changes minimal and scoped. Do not touch unrelated files or unrelated
projects.
