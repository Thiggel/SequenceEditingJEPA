# Experiment Plan

Last updated: 2026-05-29 12:38 CEST

The active backlog now lives in `../sequence-editing-report/BACKLOG.md`.

## Active Experiment

Grid 3A Sudoku local-edit ablation:

| Run | Prediction | Loss | Status |
| --- | --- | --- | --- |
| `sudoku_jepa_5m_local_direct_uniform` | direct next latent | uniform | Running as `3674778_0`, latest step `4000` |
| `sudoku_jepa_5m_local_direct_weighted` | direct next latent | changed cell high, row/col/block medium | Running as `3674778_1`, latest step `4000` |
| `sudoku_jepa_5m_local_residual_weighted` | `z_next = z_current + delta` | same weighted loss | Running as `3674778_2`, latest step `3000` |
| `sudoku_jepa_5m_local_direct_changed_only` | direct next latent | changed-cell token only | Running as `3674778_3`, latest step `3000` |

Dependent diagnostics `3674779_[0-3]` will compare single-oracle rank,
`goal_rank`, latent drift, and planning traces.

## Gate

After diagnostics finish:

1. If direct local injection remains strong under `goal_rank` and drift
   diagnostics, carry the best direct variant to rollout `N=2`.
2. If residual catches up and reduces drift, include residual in the rollout
   follow-up.
3. If changed-only remains poor, do not use changed-cell-only loss except as a
   diagnostic negative control.
4. Do not move to Maze or size sweeps until Sudoku Grid 3A final diagnostics are
   interpreted in the report repo.
