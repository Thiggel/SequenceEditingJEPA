# Experiment Plan

Last updated: 2026-05-29 21:40 CEST

The active backlog now lives in `../sequence-editing-report/BACKLOG.md`.

## Active Experiment

Grid 3A Sudoku local-edit ablation:

| Run | Prediction | Loss | Status |
| --- | --- | --- | --- |
| `sudoku_jepa_5m_local_direct_uniform` | direct next latent | uniform | Completed as `3674778_0`, step `5000`, online solve `1.0 / 1.0 / 1.0` |
| `sudoku_jepa_5m_local_direct_weighted` | direct next latent | changed cell high, row/col/block medium | Completed as `3674778_1`, step `5000`, online solve `1.0 / 1.0 / 1.0` |
| `sudoku_jepa_5m_local_residual_weighted` | `z_next = z_current + delta` | same weighted loss | Completed as `3674778_2`, step `5000`, online solve `0.0 / 0.0 / 0.0` |
| `sudoku_jepa_5m_local_direct_changed_only` | direct next latent | changed-cell token only | Completed as `3674778_3`, step `5000`, online solve `0.0 / 0.0 / 0.0` |

Dependent diagnostics `3674779_[0-3]` failed on CLI argument formatting before
model load. The wrapper was fixed and diagnostics were resubmitted as
`3676904_[0-3]`; they completed successfully.

## Gate

Grid 3A diagnostic decision:

1. Direct local injection passes the action-grounding gate: direct uniform and
   direct weighted both have diagnostic `goal_rank` mean/top1 `1.0`.
2. Direct weighted is the preferred follow-up seed: it has lower short drift
   than uniform and better terminal-planning proximity, despite slightly worse
   single-oracle rank.
3. Residual is rejected for the next branch because rollout drift explodes
   (`drift@20 103`, terminal `1940`).
4. Changed-cell-only loss is rejected except as a negative control because
   `goal_rank` and planning are poor.
5. Next safe experiment: a short Sudoku local-direct weighted rollout `N=2` to
   target the remaining long-horizon drift / closed-loop exactness bottleneck.
   Do not move to Maze or size sweeps until that local rollout is diagnosed.
