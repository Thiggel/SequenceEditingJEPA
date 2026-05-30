# Results

Last updated: 2026-05-30 13:31 CEST

Detailed results now live in `../sequence-editing-report/RESULTS.md` and the
ongoing LaTeX report `../sequence-editing-report/report.tex`.

## Current Key Result

Grid 3B large diagnostics now isolate the lead checkpoint's failure mode. The
same `sudoku_jepa_5m_local_direct_weighted` model that has perfect sampled
`goal_rank` still cannot solve under closed-loop latent rollout planning
(`0/64` exact solves), but re-encoded symbolic-state planning solves all 64
diagnostic boards. Under oracle-goal diagnostics, the remaining failure is
latent rollout drift / stale latent state rather than the local action scorer.

| Planner | Scoring | Solve | Terminal | Mean remaining Hamming |
| --- | --- | ---: | ---: | ---: |
| latent rollout | step energy | 0.0 | 0.015625 | 4.734375 |
| latent rollout | terminal energy | 0.0 | 0.0625 | 4.671875 |
| re-encoded state | step energy | 1.0 | 1.0 | 0.0 |
| re-encoded state | terminal energy | 1.0 | 1.0 | 0.0 |

Large-diagnostic action grounding remains strong: `goal_rank` mean/top1 is
`1.0` over 4096 sampled states, while the stricter single-oracle rank mean is
`21.59`. Latent drift still jumps from `0.079` at 10 oracle steps to `1.742` at
20 steps and about `2.0` near terminal states.

Generated artifacts: `../sequence-editing-report/assets/grid3b/` contains the
planning comparison, drift curve, terminal mismatch heatmap, rollout `N=2`
training-so-far curve, CSV tables, and concrete latent failure examples.

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

Grid 3B rollout `N=2` is still running as job `3680020`. At 13:26 CEST it had
written `checkpoint-3000.pt` and `checkpoint.pt`; online metrics were eval loss
`0.000186`, oracle mean rank `17.0625`, and H1/H2/H4 solve
`1.0 / 1.0 / 1.0`. The dependent diagnostics job `3680021` remains pending on
`afterok:3680020` and is the next decisive read. Current oversight `3680033` is
running, and exactly one successor, `3680652`, is pending for
`2026-05-30 17:25:44 CEST`.

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
