# Runbook

Last updated: 2026-05-29 17:41 CEST

Long-form handoff source of truth: `../sequence-editing-report`.

- Ongoing LaTeX report: `../sequence-editing-report/report.tex`
- Experiment backlog: `../sequence-editing-report/BACKLOG.md`
- Live status: `../sequence-editing-report/STATUS.md`
- Results and insights: `../sequence-editing-report/RESULTS.md`
- Chronological log: `../sequence-editing-report/LOG.md`

## Environment

```bash
source scripts/env.sh
python -m pytest -q tests
```

Runtime outputs default to:

```text
/home/vault/$(id -gn)/$USER/sequence-editing
```

## Active Slurm Snapshot

| Job | State | Notes |
| --- | --- | --- |
| `3674778_[0-3]` | COMPLETED | Grid 3A training complete; all four roots have `metrics.json` and `checkpoint.pt`. |
| `3674779_[0-3]` | FAILED | First Grid 3A diagnostics failed before model load: wrapper passed comma-separated `--horizons`. |
| `3676904_[0-3]` | RUNNING | Resubmitted Grid 3A diagnostics after wrapper fix; started `2026-05-29 17:41:20 CEST`. |
| `3675734` | RUNNING | Current oversight job; next begin-time oversight `3676879` is pending. |
| `3676879` | PENDING | Recurring oversight, begin time `2026-05-29 21:36:14 CEST`. |

Check live state:

```bash
squeue -j 3674778,3674779,3675734,3676879,3676904 -o "%.18i %.9T %.28j %.10M %.20S %R"
sacct -j 3674778,3674779,3675734,3676879,3676904 --format=JobID,JobName%30,State,ExitCode,Elapsed,Start,End,NodeList
```

## Current Operational Read

Grid 3A training finished. The direct local-injection variants retained online
H1/H2/H4 solve `1.0` at step `5000`; residual and changed-only variants stayed
at solve `0.0`. Treat this as an online-eval result only until resubmitted
diagnostics `3676904_[0-3]` finish and `goal_rank`/drift/planning traces are
interpreted. Do not start 10M/20M sweeps or Maze follow-ups before that gate.
