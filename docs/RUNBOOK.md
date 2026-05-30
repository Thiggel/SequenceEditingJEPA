# Runbook

Last updated: 2026-05-30 13:31 CEST

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
| `3680020` | RUNNING | Grid 3B `local_direct_weighted` rollout `N=2`; latest checkpoint at step `3000` as of 13:26 CEST. |
| `3680021` | PENDING | Dependent Grid 3B rollout `N=2` diagnostics, `afterok:3680020`. |
| `3676879` | COMPLETED | Previous puzzle oversight completed at `2026-05-29 21:48:10 CEST`. |
| `3677391` | COMPLETED | Previous puzzle oversight completed at `2026-05-30 01:43:35 CEST`. |
| `3678050` | COMPLETED | Previous puzzle oversight completed at `2026-05-30 05:48:59 CEST`. |
| `3679094` | COMPLETED | Previous puzzle oversight completed at `2026-05-30 09:44:26 CEST`. |
| `3679877` | CANCELLED | Stale pending oversight cancelled after replacing the prompt with the enhanced template. |
| `3680033` | RUNNING | Enhanced recurring oversight, started `2026-05-30 13:24:55 CEST`. |
| `3680652` | PENDING | Next enhanced recurring oversight, begin time `2026-05-30 17:25:44 CEST`. |

Check live state:

```bash
squeue -j 3680019,3680020,3680021,3680033,3680652 -o "%.18i %.9T %.28j %.10M %.20S %R"
sacct -j 3680019,3680020,3680021,3680033,3680652 --format=JobID,JobName%30,State,ExitCode,Elapsed,Start,End,NodeList
```

## Current Operational Read

Grid 3B large diagnostics completed for the current lead checkpoint. Latent
rollout planning still has exact solve `0.0` on 64 examples, with mean remaining
Hamming `4.734` under step-energy scoring and `4.672` under terminal-only
scoring. Re-encoded symbolic-state planning solved all 64 examples under both
scoring modes (`solve_rate=1.0`, mean remaining Hamming `0.0`). Terminal-only
scoring therefore does not change the conclusion; it only raises the latent
filled-board terminal rate from `1/64` to `4/64`.

Interpretation: under oracle-goal diagnostics, action scoring is sufficient when
candidate symbolic states are re-encoded exactly. The lead failure is latent
rollout drift / stale latent state, not the local action scorer. This is not a
deployable solve metric because the diagnostic still uses the oracle goal state
and because the latent and re-encoded planners sampled separate eval examples,
but the `0/64` versus `64/64` split is large enough to set the current gate.

Concrete latent terminal errors are mostly a few blank or wrong cells: the
terminal-energy latent records have 299 mismatches across 64 boards, including
189 blanks left as `0` and 110 wrong nonzero values. Hotspots are concentrated
in columns `8`, `2`, and `7` and rows `3`, `7`, `2`, `0`, and `5`. Generated
analysis artifacts live under `../sequence-editing-report/assets/grid3b/`.

Grid 3B rollout `N=2` remains active. At 13:26 CEST, job `3680020` had written
`checkpoint-3000.pt` and `checkpoint.pt`; online metrics were eval loss
`0.000186`, oracle mean rank `17.0625`, and H1/H2/H4 solve `1.0 / 1.0 / 1.0`.
Do not accept that as final solve quality: wait for dependent diagnostics
`3680021` after training completes, then compare `goal_rank`, drift, latent vs
re-encoded planning, terminal solve, remaining Hamming, mismatch concentration,
and training curves against Grid 3A direct weighted.

Partition housekeeping at 13:26 CEST: `3680020` and `3680033` are already
running on `a100`; `3680021` is dependency-blocked and `3680652` is
begin-time-blocked, so no `scontrol update ... Partition=...` was applied.

Oversight now uses `scripts/oversight/puzzle_oversight_prompt.md`. That prompt
requires each run to reconcile Slurm/artifacts with the backlog, inspect
concrete planner examples, question assumptions, add useful report figures and
tables, fix/resubmit small failures, and keep the four-hour oversight chain
alive.
