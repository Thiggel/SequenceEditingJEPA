# Results

Last updated: 2026-05-29 12:38 CEST

Detailed results now live in `../sequence-editing-report/RESULTS.md` and the
ongoing LaTeX report `../sequence-editing-report/report.tex`.

## Current Key Result

Grid 3A local value-only action injection is running and already strongly
outperforms the old global-broadcast action conditioning on the small online
Sudoku planning eval.

| Run | Step | Eval loss | Mean rank | H1/H2/H4 solve |
| --- | ---: | ---: | ---: | --- |
| `sudoku_jepa_5m_local_direct_uniform` | 4000 | 0.000235 | 14.8125 | 1.0 / 1.0 / 1.0 |
| `sudoku_jepa_5m_local_direct_weighted` | 4000 | 0.000100 | 13.15625 | 1.0 / 1.0 / 1.0 |
| `sudoku_jepa_5m_local_residual_weighted` | 3000 | 0.00538 | 33.75 | 0.0 / 0.0 / 0.0 |
| `sudoku_jepa_5m_local_direct_changed_only` | 3000 | 0.12825 | 232.25 | 0.0 / 0.0 / 0.0 |

Treat this as preliminary until dependent diagnostics `3674779_[0-3]` complete.
Grid 3A is still running; these values are from logs/`metrics.jsonl`, and final
`metrics.json` files are not expected until training exits.

## Prior Read

- Grid 1 diagnostics showed true re-encoded oracle states were monotonic toward
  the goal, while predicted latent rollouts drifted badly.
- Grid 2A rollout training improved 10/20-step drift but worsened action rank
  and did not fix terminal planning.
- The current hypothesis is that local action injection fixes a major action
  grounding failure caused by global action broadcast.
