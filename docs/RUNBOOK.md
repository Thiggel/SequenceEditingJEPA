# Runbook

Last updated: 2026-05-30 17:31 CEST

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

This table is restricted to jobs submitted from this repo's active puzzle-JEPA
Slurm wrappers. Other user-account HFSA/paired arrays are not part of this
repo snapshot.

| Job | State | Notes |
| --- | --- | --- |
| `3664581_[0-1]` | COMPLETED | Grid 0 smoke; infrastructure only, solve remained `0.0`. |
| `3665018_[0-4]` | COMPLETED | Grid 1 one-step/curriculum runs; planning solve remained `0.0`. |
| `3667044_[0-4]` | COMPLETED | Replacement Grid 1 diagnostics completed after earlier cancelled stale diagnostics. |
| `3671344_[0-3]` | COMPLETED | Grid 2A rollout `N=2/4` for Sudoku/Maze; no terminal solve. |
| `3671345_[0-3]` | FAILED | First Grid 2A diagnostics failed on stale CLI flags; superseded by `3673400_[0-3]`. |
| `3673400_[0-3]` | COMPLETED | Replacement Grid 2A diagnostics completed. |
| `3674778_[0-3]` | COMPLETED | Grid 3A training complete; all four roots have `metrics.json` and `checkpoint.pt`. |
| `3674779_[0-3]` | FAILED | First Grid 3A diagnostics failed before model load: wrapper passed comma-separated `--horizons`. |
| `3676904_[0-3]` | COMPLETED | Resubmitted Grid 3A diagnostics completed; all four roots have `diagnostics/diagnostics.json`. |
| `3680019` | COMPLETED | Grid 3B large diagnostics for `sudoku_jepa_5m_local_direct_weighted`; latent rollout solve `0.0`, re-encoded symbolic-state planning solve `1.0`. |
| `3680020` | COMPLETED | Grid 3B `local_direct_weighted` rollout `N=2`; final checkpoint step `5000`, exit `0:0`. |
| `3680021` | COMPLETED | Dependent Grid 3B rollout `N=2` diagnostics; latent terminal-energy solve `4/64`, re-encoded planning `64/64`. |
| `3676879` | COMPLETED | Previous puzzle oversight completed at `2026-05-29 21:48:10 CEST`. |
| `3677391` | COMPLETED | Previous puzzle oversight completed at `2026-05-30 01:43:35 CEST`. |
| `3678050` | COMPLETED | Previous puzzle oversight completed at `2026-05-30 05:48:59 CEST`. |
| `3679094` | COMPLETED | Previous puzzle oversight completed at `2026-05-30 09:44:26 CEST`. |
| `3679877` | CANCELLED | Stale pending oversight cancelled after replacing the prompt with the enhanced template. |
| `3680033` | COMPLETED | Enhanced recurring oversight completed at `2026-05-30 13:41:13 CEST`. |
| `3680652` | RUNNING | Current enhanced recurring oversight, started `2026-05-30 17:26:00 CEST`. |
| `3681711` | PENDING | Next enhanced recurring oversight, begin time `2026-05-30 21:26:03 CEST`. |

Check live state:

```bash
squeue -j 3680019,3680020,3680021,3680033,3680652,3681711 -o "%.18i %.9T %.28j %.10M %.20S %R"
sacct -j 3680019,3680020,3680021,3680033,3680652,3681711 --format=JobID,JobName%30,State,ExitCode,Elapsed,Start,End,NodeList
```

## Current Operational Read

Grid 3B rollout `N=2` completed and improved proximity, but it did not satisfy
the exact-solve gate. Final online metrics at step `5000` were eval loss
`0.000138`, oracle mean rank `12.34375`, and H1/H2/H4 solve
`1.0 / 1.0 / 1.0`; the larger diagnostics are the meaningful read. Job
`3680021` reports latent terminal-energy planning solve `4/64`, filled-board
terminal rate `26/64`, and mean remaining Hamming `2.453`. Re-encoded
symbolic-state planning still solves `64/64`.

Compared with the lead large diagnostics (`3680019`), rollout `N=2` cuts
terminal-energy mismatches from 299 to 157 and reduces drift at 10/20 oracle
steps from `0.079/1.742` to `0.041/1.495`. Terminal weighted latent drift is
not fixed (`2.014 -> 2.163`), so the main interpretation still holds: under
oracle-goal diagnostics, the action scorer is sufficient when candidate boards
are re-encoded, while stale latent rollout remains the bottleneck. The result
is not deployable because it still uses the oracle goal and because latent and
re-encoded planner samples are not paired example-by-example.

Planning intentionally allows overwrites/conflicts on mutable Sudoku cells for
diagnosis, so terminal boards can be fully filled but wrong. Treat online
H1/H2/H4 and terminal fill rate as sanity checks, not solve quality. Generated
analysis artifacts live under `../sequence-editing-report/assets/grid3b/`,
including rollout `N=2` planning, drift, remaining-Hamming, mismatch heatmap,
training curve, and concrete terminal board examples.

Partition housekeeping at 17:31 CEST: only repo job `3680652` is running, and
the only later repo job `3681711` is begin-time-blocked, so no
`scontrol update ... Partition=...` was useful.

Oversight now uses `scripts/oversight/puzzle_oversight_prompt.md`. That prompt
requires each run to reconcile Slurm/artifacts with the backlog, inspect
concrete planner examples, question assumptions, add useful report figures and
tables, fix/resubmit small failures, and keep the four-hour oversight chain
alive. The next safe experiment is a small reset/re-encoding diagnostic or
ablation, not Maze or broad capacity sweeps.
