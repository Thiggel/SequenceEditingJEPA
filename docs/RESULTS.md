# Results

Last updated: 2026-05-31 01:31 CEST

Detailed results now live in `../sequence-editing-report/RESULTS.md` and the
ongoing LaTeX report `../sequence-editing-report/report.tex`.

## Current Key Result

Grid 3B rollout `N=2` gives a real but insufficient improvement. It preserves
sampled action grounding (`goal_rank` mean/top1 `1.0`) and improves latent
terminal-energy planning from `0/64` to `4/64` solves, with mean remaining
Hamming falling from `4.672` to `2.453`. Re-encoded symbolic-state planning
still solves `64/64`, so the bottleneck remains stale latent rollout rather
than the local action scorer under oracle-goal diagnostics.

| Run | Planner | Scoring | Solve | Terminal | Mean remaining Hamming |
| --- | --- | --- | ---: | ---: | ---: |
| lead large | latent rollout | terminal energy | 0.0 | 0.0625 | 4.671875 |
| rollout `N=2` | latent rollout | terminal energy | 0.0625 | 0.40625 | 2.453125 |
| lead large | re-encoded state | terminal energy | 1.0 | 1.0 | 0.0 |
| rollout `N=2` | re-encoded state | terminal energy | 1.0 | 1.0 | 0.0 |

Rollout `N=2` reduces drift at 10/20 oracle steps from `0.079/1.742` to
`0.041/1.495`, but terminal weighted drift is still about `2.16`. This keeps
Maze, 10M/20M sweeps, and broad controls blocked. The next safe diagnostic is a
small periodic re-encoding / latent reset branch.

Generated artifacts: `../sequence-editing-report/assets/grid3b/` contains the
lead and rollout `N=2` planning comparisons, drift curves, terminal
remaining-Hamming distributions, mismatch heatmaps, final training curve, CSV
tables, and concrete latent terminal examples.

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

Grid 3C reset-cadence diagnostics were implemented and submitted as `3682924`
for the rollout `N=2` checkpoint. The job compares latent no-reset, reset every
2/4/8/16 actions, and full re-encoded planning on the same 64 sampled boards,
writing `diagnostics_reset_cadence/` at completion. As of `2026-05-31 01:31
CEST`, it is still running with empty stderr and no reset-cadence files yet.
Current oversight `3682864` is running, and exactly one successor, `3683472`,
is pending for `2026-05-31 05:27:01 CEST`.

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
