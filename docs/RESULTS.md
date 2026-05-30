# Results

Last updated: 2026-05-30 09:50 CEST

Detailed results now live in `../sequence-editing-report/RESULTS.md` and the
ongoing LaTeX report `../sequence-editing-report/report.tex`.

## Current Key Result

Grid 3A local value-only action injection finished and diagnostics now confirm
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
