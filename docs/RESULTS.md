# Results

Last updated: 2026-05-31 09:30 CEST

Detailed results now live in `../sequence-editing-report/RESULTS.md` and the
ongoing LaTeX report `../sequence-editing-report/report.tex`.

## Current Key Result

Grid 3C reset-cadence diagnostics show that periodic planner-state re-encoding
recovers the re-encoded oracle-goal result. On paired 64-board samples for the
rollout `N=2` checkpoint, no-reset terminal-energy planning solved `2/64`, reset
every 2/4 actions solved `64/64` under both step and terminal energy, reset
every 8/16 actions solved `64/64` under terminal energy, and full re-encoded
planning solved `64/64`.

| Run | Planner | Scoring | Solve | Terminal | Mean remaining Hamming |
| --- | --- | --- | ---: | ---: | ---: |
| lead large | latent rollout | terminal energy | 0.0 | 0.0625 | 4.671875 |
| rollout `N=2` | latent rollout | terminal energy | 0.0625 | 0.40625 | 2.453125 |
| Grid 3C paired | no reset | terminal energy | 0.03125 | 0.5625 | 2.265625 |
| Grid 3C paired | reset every 4 | step energy | 1.0 | 1.0 | 0.0 |
| Grid 3C paired | reset every 8 | terminal energy | 1.0 | 1.0 | 0.0 |
| lead large | re-encoded state | terminal energy | 1.0 | 1.0 | 0.0 |
| rollout `N=2` | re-encoded state | terminal energy | 1.0 | 1.0 | 0.0 |

This passes the mechanism gate for a planner-state reset/re-encoding branch,
but it is still an oracle-goal diagnostic, not a deployable solver. Maze,
10M/20M sweeps, and broad controls remain blocked until the reset branch is
confirmed and integrated.

Generated artifacts: `../sequence-editing-report/assets/grid3b/` contains the
lead and rollout `N=2` planning comparisons, drift curves, terminal
remaining-Hamming distributions, mismatch heatmaps, final training curve, CSV
tables, Grid 3C reset-cadence plots/CSVs, and concrete paired examples.

## Grid 3A Grounding Result

Grid 3A local value-only action injection finished and diagnostics confirmed
the main action-grounding result. Direct local injection strongly outperformed
the old global-broadcast action conditioning; both direct variants rank a
goal-correct action first on every sampled diagnostic state.

| Run | Step | Eval loss | Mean rank | H1/H2/H4 solve |
| --- | ---: | ---: | ---: | --- |
| `sudoku_jepa_5m_local_direct_uniform` | 5000 | 0.000187 | 15.96875 | 1.0 / 1.0 / 1.0 |
| `sudoku_jepa_5m_local_direct_weighted` | 5000 | 0.0000639 | 16.25 | 1.0 / 1.0 / 1.0 |
| `sudoku_jepa_5m_local_residual_weighted` | 5000 | 0.00234 | 115.96875 | 0.0 / 0.0 / 0.0 |
| `sudoku_jepa_5m_local_direct_changed_only` | 5000 | 0.11818 | 246.53125 | 0.0 / 0.0 / 0.0 |

All four roots have final `metrics.json`, `metrics.jsonl`, and `checkpoint.pt`.
The first dependent diagnostics array `3674779_[0-3]` failed because the wrapper
passed comma-separated `--horizons`; after a local fix and smoke test,
diagnostics were resubmitted as `3676904_[0-3]` and completed.

`H1/H2/H4 solve` is an online training metric over only 8 eval examples. It is
not the final solver metric: H1 scores legal one-step actions by predicted
next-latent distance to the goal latent; H2/H4 expand exact symbolic board
states for a short horizon and re-encode candidate terminal states. Treat
diagnostic terminal planning as the stricter Sudoku-solve read.

## Active Follow-Up

Grid 3B rollout `N=2` completed as `3680020` and diagnostics completed as
`3680021`. Final online metrics at step `5000` were eval loss `0.000138`,
oracle mean rank `12.34375`, and H1/H2/H4 solve `1.0 / 1.0 / 1.0`, but the
larger diagnostics show exact latent solve remains weak.

Grid 3C reset-cadence diagnostics completed as `3682924` and wrote
`diagnostics_reset_cadence/diagnostics.json` plus paired reset planning records.
The follow-up Grid 3D reset-large confirmation is running as `3683903`; it
compares no reset, reset every 4/8 actions, and full re-encoded planning on 128
paired boards, writing `diagnostics_reset_cadence_large/` only at completion.
As of 09:30 CEST, stderr is empty and the output directory has not been created.
Oversight `3683472` completed cleanly, current oversight `3683863` is running,
and exactly one successor, `3684237`, is pending for `2026-05-31 13:27:20 CEST`.

## Grid 3A Diagnostics

| Run | Goal-rank mean / top1 | Rank mean | Drift @10 / @20 / terminal | Terminal planning |
| --- | ---: | ---: | ---: | --- |
| `local_direct_uniform` | 1.0 / 1.0 | 18.42 | 0.119 / 1.788 / 2.060 | solve 0.0, terminal 0.0, remaining Hamming 5.625 |
| `local_direct_weighted` | 1.0 / 1.0 | 21.82 | 0.078 / 1.728 / 2.007 | solve 0.0, terminal 0.125, remaining Hamming 4.25 |
| `local_residual_weighted` | 2.085 / 0.493 | 124.20 | 2.761 / 103.3 / 1940 | solve 0.0, terminal 0.0, remaining Hamming 47.375 |
| `local_direct_changed_only` | 15.49 / 0.0566 | 242.79 | 1.818 / 1.864 / 1.883 | solve 0.0, terminal 0.0, remaining Hamming 54.375 |

Against Sudoku Grid 1 and Grid 2A, direct local injection greatly improves
action grounding and closed-loop proximity: Grid 1 mix50/50 had rank mean
`167.68` and remaining Hamming `51.75`, while Grid 2A `N=4` had rank mean
`209.14` and remaining Hamming `53.875`. Local direct weighted reduces the
closed-loop miss to about four cells, but long-horizon drift remains high
(`@20 1.728`, terminal `2.007`) and terminal solve is still `0.0`.

## Prior Read

- Grid 1 diagnostics showed true re-encoded oracle states were monotonic toward
  the goal, while predicted latent rollouts drifted badly.
- Grid 2A rollout training improved 10/20-step drift but worsened action rank
  and did not fix terminal planning.
- Local action injection fixes a major action grounding failure caused by global
  action broadcast. The remaining bottleneck is long-horizon drift /
  closed-loop exactness after locally grounded one-step predictions.
- The residual/delta variant is not currently a win. It predicts an additive
  correction to a contextual latent and accumulates errors under closed-loop
  rollout; its drift explodes by 20/terminal steps.
