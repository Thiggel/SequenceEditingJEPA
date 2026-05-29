# Runbook

Last updated: 2026-05-29 12:30 CEST

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
| `3674778_0` | RUNNING | Grid 3A `sudoku_jepa_5m_local_direct_uniform`; latest step `4000`, H1/H2/H4 solve `1.0`. |
| `3674778_1` | RUNNING | Grid 3A `sudoku_jepa_5m_local_direct_weighted`; latest step `4000`, H1/H2/H4 solve `1.0`. |
| `3674778_2` | RUNNING | Grid 3A `sudoku_jepa_5m_local_residual_weighted`; latest step `3000`, H1/H2/H4 solve `0.0`. |
| `3674778_3` | RUNNING | Grid 3A `sudoku_jepa_5m_local_direct_changed_only`; latest step `2000`, H1/H2/H4 solve `0.0`. |
| `3674779_[0-3]` | PENDING | Dependent Grid 3A diagnostics, `afterok:3674778`. |
| `3674780` | PENDING | Updated recurring oversight, begin time `2026-05-29 12:34:53 CEST`. |

Check live state:

```bash
squeue -j 3674778,3674779,3674780 -o "%.18i %.9T %.28j %.10M %.20S %R"
sacct -j 3674778,3674779,3674780 --format=JobID,JobName%30,State,ExitCode,Elapsed,Start,End,NodeList
```

## Current Operational Read

Grid 3A is the active branch. Local value-only action injection is the first
clearly positive JEPA planning signal. Do not start 10M/20M sweeps until
`3674779_[0-3]` diagnostics finish and the report backlog is updated.
