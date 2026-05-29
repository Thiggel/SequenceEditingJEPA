# Runbook

Last updated: 2026-05-29 21:40 CEST

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
| `3676904_[0-3]` | COMPLETED | Resubmitted Grid 3A diagnostics completed; all four roots have `diagnostics/diagnostics.json`. |
| `3675734` | COMPLETED | Previous puzzle oversight. |
| `3676879` | RUNNING | Current puzzle oversight, started `2026-05-29 21:36:21 CEST`. |
| `3677391` | PENDING | Recurring oversight, begin time `2026-05-30 01:36:23 CEST`. |

Check live state:

```bash
squeue -j 3674778,3674779,3675734,3676879,3676904,3677391 -o "%.18i %.9T %.28j %.10M %.20S %R"
sacct -j 3674778,3674779,3675734,3676879,3676904,3677391 --format=JobID,JobName%30,State,ExitCode,Elapsed,Start,End,NodeList
```

## Current Operational Read

Grid 3A diagnostics finished. Direct local value injection fixed sampled
goal-action grounding: both direct variants have `goal_rank` mean/top1 `1.0`.
Direct weighted is the current lead because it has lower short drift than
uniform (`drift@10 0.078` vs `0.119`) and better closed-loop terminal planning
proximity (`terminal_rate 0.125`, mean remaining Hamming `4.25` vs `5.625`),
though terminal solve remains `0.0`.

Residual prediction and changed-cell-only loss are rejected for the next branch:
residual has explosive rollout drift (`drift@20 103`, terminal `1940`), and
changed-only has poor goal rank (`15.49`) plus poor planning. The concrete
bottleneck is now long-horizon drift / closed-loop exactness after strong local
one-step grounding. Next safe experiment is a short local-direct weighted
rollout `N=2`; do not start Maze, 10M/20M sweeps, or broad controls before that
follow-up is implemented, run, and diagnosed.
