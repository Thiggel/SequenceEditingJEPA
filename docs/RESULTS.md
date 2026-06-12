# Results

Last updated: 2026-06-12 11:30 CEST

Detailed historical results live in `../sequence-editing-report/RESULTS.md` and
`../sequence-editing-report/report.tex`.

## Current Result

No Grid 5 result yet. Grid 5 `3722613_[0-23]` is running.

## Current Interpretation

The old tokenized oracle-goal reset branch solved Sudoku, but that result was
heavily supported by a cell-factorized latent representation and oracle solved
board latents. The new Grid 5 branch tests whether a compact single-state JEPA
can learn a useful metric geometry when SIGReg is always part of the objective.

The old global-MLP branch did not include this SIGReg geometry regularizer, so
it is no longer treated as a decisive test of single-state JEPA.

## Diagnostics To Read First

For each Grid 5 run, read:

- `diagnostics/diagnostics.json`
- `diagnostics/trajectory_records.jsonl`
- `diagnostics/action_rank_records.jsonl`
- `diagnostics/action_rank_examples.jsonl`

Primary success signals before exact solve:

- latent `std_mean` near `1`, healthy `pairwise_distance_mean`, low
  `cov_offdiag_abs_mean`
- high `latent_monotone_rate`
- low `latent_gold_rank_mean` and high `latent_top_goal_value_rate`
- low learned `goal_energy_abs_error_mean`
- learned-energy action ranking close to oracle latent ranking
