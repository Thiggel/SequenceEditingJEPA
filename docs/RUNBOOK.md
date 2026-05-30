# Runbook

Last updated: 2026-05-30 10:25 CEST

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
| `3680019` | RUNNING | Grid 3B large diagnostics for `sudoku_jepa_5m_local_direct_weighted`; writes `diagnostics_large/`, including latent and re-encoded planning records. |
| `3680020` | RUNNING | Grid 3B `local_direct_weighted` rollout `N=2`; output root `sudoku_jepa_5m_local_direct_weighted_rollout_n2`. |
| `3680021` | PENDING | Dependent Grid 3B rollout `N=2` diagnostics, `afterok:3680020`. |
| `3676879` | COMPLETED | Previous puzzle oversight completed at `2026-05-29 21:48:10 CEST`. |
| `3677391` | COMPLETED | Previous puzzle oversight completed at `2026-05-30 01:43:35 CEST`. |
| `3678050` | COMPLETED | Previous puzzle oversight completed at `2026-05-30 05:48:59 CEST`. |
| `3679094` | COMPLETED | Previous puzzle oversight completed at `2026-05-30 09:44:26 CEST`. |
| `3679877` | CANCELLED | Stale pending oversight cancelled after replacing the prompt with the enhanced template. |
| `3680033` | PENDING | Enhanced recurring oversight, begin time `2026-05-30 13:24:28 CEST`. |

Check live state:

```bash
squeue -j 3680019,3680020,3680021,3680033 -o "%.18i %.9T %.28j %.10M %.20S %R"
sacct -j 3680019,3680020,3680021,3679094,3679877,3680033 --format=JobID,JobName%30,State,ExitCode,Elapsed,Start,End,NodeList
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
one-step grounding.

Grid 3B is now active. Job `3680019` runs larger diagnostics on the current
weighted direct checkpoint and compares latent rollout planning against
re-encoded symbolic-state planning. Job `3680020` trains the short
local-direct weighted rollout `N=2`; dependent job `3680021` will run the same
larger diagnostics on the rollout checkpoint if training succeeds. Do not start
Maze, 10M/20M sweeps, or broad controls until these finish.

Live check at 10:18 CEST: the two running jobs have no stderr output yet.
`3680020` has created its run `config.json`; diagnostics output for `3680019`
has not been written yet.

Oversight now uses `scripts/oversight/puzzle_oversight_prompt.md`. That prompt
requires each run to reconcile Slurm/artifacts with the backlog, inspect
concrete planner examples, question assumptions, add useful report figures and
tables, fix/resubmit small failures, and keep the four-hour oversight chain
alive.
