# Current Experiments

Last updated: 2026-06-30 18:03 CEST

## Old-Local Fast Wave

This sweep tests whether the old Sudoku action interface, `old_local_value`
(inject the digit/value embedding into the edited cell), recovers the strong
oracle-goal planning signal. All runs use EMA+VICReg, temporal straightening,
predicted-goal progress monotonicity, `q(c,H0,Ht)` goal conditioning, LR
`1e-4`, batch `8`, no grad accumulation, and `5000` optimizer steps.

Slurm:

| Array | State | Notes |
|---|---:|---|
| `3797928` train `0-17` | completed | all 18 checkpoints written |
| `3797929` eval `0-17` | running | all 18 tasks running on `rtxpro6k`; `.err` files empty |

Eval is still partial: `373 / 1984` expected planner rows are written
(`17.7%`). So far only `mpc_beam + symbolic_reencode` rows have appeared; no
latent-rollout or hierarchical-beam rows have been reached yet.

## Variants

| Group | Variants | Meaning |
|---|---|---|
| Dense horizon | `dense_k1,k4,k8,k16,k32` | supervise every rollout step up to that horizon, no hierarchy |
| Hierarchy | `hier_l4`, `hier_l4_l16`, `hier_l4_l16_l32`, `hier_l4_l16_shared`, `hier_l4_l16_hier_dense` | hierarchy depth/parameter sharing/dense high-level supervision |
| Ranking | `rank_*` | progress/action ranking target or mode ablations on top of `[4,16]` hierarchy |

## Training

| Variant | Params | Final loss | Dyn | Dense | Hier | Progress | Action |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense_k1 | 27.9M | 1.728 | 0.016 | 0.000 | 0.000 | 0.124 | 1.326 |
| dense_k4 | 27.9M | 1.711 | 0.014 | 0.010 | 0.000 | 0.122 | 1.347 |
| dense_k8 | 27.9M | 1.653 | 0.015 | 0.010 | -0.000 | 0.123 | 1.306 |
| dense_k16 | 27.9M | 1.714 | 0.017 | 0.010 | -0.000 | 0.124 | 1.313 |
| dense_k32 | 27.9M | 1.622 | 0.019 | 0.008 | 0.000 | 0.123 | 1.282 |
| hier_l4 | 33.2M | 1.673 | 0.013 | 0.007 | 0.007 | 0.118 | 1.351 |
| hier_l4_l16 | 37.5M | 1.696 | 0.014 | 0.008 | 0.011 | 0.121 | 1.344 |
| hier_l4_l16_l32 | 41.8M | 1.594 | 0.018 | 0.007 | 0.016 | 0.121 | 1.270 |
| hier_l4_l16_shared | 33.2M | 1.664 | 0.014 | 0.007 | 0.009 | 0.122 | 1.306 |
| hier_l4_l16_hier_dense | 37.5M | 1.636 | 0.014 | 0.008 | 0.016 | 0.120 | 1.305 |
| rank_oracle_progress | 37.5M | 1.695 | 0.015 | 0.008 | 0.012 | 0.123 | 1.317 |
| rank_both_progress | 37.5M | 1.728 | 0.016 | 0.008 | 0.013 | 0.128 | 1.351 |
| rank_no_progress | 37.5M | 1.505 | 0.015 | 0.009 | 0.012 | 0.000 | 1.313 |
| rank_pairwise_oracle_action | 37.5M | 1.511 | 0.014 | 0.007 | 0.011 | 0.128 | 1.149 |
| rank_pairwise_both_action | 37.5M | 1.611 | 0.012 | 0.006 | 0.009 | 0.127 | 1.277 |
| rank_listwise_pred_action | 37.5M | 3.527 | 0.023 | 0.013 | 0.017 | 0.148 | 3.015 |
| rank_listwise_both_action | 37.5M | 3.679 | 0.028 | 0.015 | 0.021 | 0.136 | 3.014 |
| rank_no_action | 37.5M | 0.167 | 0.007 | 0.004 | 0.004 | 0.115 | 0.000 |

## Partial Planner Results

All partial rows have `0/10` solves so far. Best rows are measured by lowest
remaining Hamming among the rows currently written.

| Variant | Rows | Expected | Best remaining Hamming | Best setting |
|---|---:|---:|---:|---|
| dense_k1 | 22 | 64 | 37.6 | oracle raw Euclidean, depth 32 |
| dense_k4 | 22 | 64 | 40.0 | oracle raw Euclidean, depth 16 |
| dense_k8 | 22 | 64 | 39.1 | oracle raw Euclidean, depth 4 |
| dense_k16 | 19 | 64 | 41.0 | oracle raw Euclidean, depth 16 |
| dense_k32 | 19 | 64 | 41.4 | oracle raw Euclidean, depth 16 |
| hier_l4 | 19 | 128 | 36.9 | oracle raw MSE, depth 16 |
| hier_l4_l16 | 19 | 128 | 39.9 | oracle raw Euclidean, depth 16 |
| hier_l4_l16_l32 | 19 | 128 | 38.1 | oracle raw MSE, depth 4 |
| hier_l4_l16_shared | 19 | 128 | 37.0 | oracle raw MSE, depth 16 |
| hier_l4_l16_hier_dense | 19 | 128 | 40.3 | oracle raw MSE, depth 32 |
| rank_oracle_progress | 19 | 128 | 14.5 | oracle raw MSE, depth 16 |
| rank_both_progress | 19 | 128 | 27.3 | oracle raw Euclidean, depth 16 |
| rank_no_progress | 19 | 128 | 36.3 | oracle raw MSE, depth 32 |
| rank_pairwise_oracle_action | 19 | 128 | 15.4 | oracle normalized distance, depth 4 |
| rank_pairwise_both_action | 19 | 128 | 25.4 | oracle normalized distance, depth 32 |
| rank_listwise_pred_action | 19 | 128 | 20.6 | oracle raw Euclidean, depth 16 |
| rank_listwise_both_action | 19 | 128 | 20.0 | oracle raw MSE, depth 32 |
| rank_no_action | 19 | 128 | 41.0 | oracle raw MSE, depth 16 |

Best rows overall so far:

| Rank | Variant | Remaining Hamming | Score | Depth |
|---:|---|---:|---|---:|
| 1 | rank_oracle_progress | 14.5 | oracle raw MSE | 16 |
| 2 | rank_oracle_progress | 14.8 | oracle raw MSE | 32 |
| 3 | rank_pairwise_oracle_action | 15.4 | oracle normalized distance | 4 |
| 4 | rank_oracle_progress | 15.5 | oracle raw MSE | 4 |
| 5 | rank_pairwise_oracle_action | 16.0 | oracle normalized distance | 16 |

## Early Diagnostics

| Variant | Oracle Spearman | Pred Spearman | Oracle top-1 | Pred top-1 | Drift h16 | Goal cosine |
|---|---:|---:|---:|---:|---:|---:|
| dense_k1 | 0.414 | 0.998 | 0.031 | 0.094 | 0.851 | 0.847 |
| dense_k4 | 0.935 | 0.997 | 0.125 | 0.094 | 0.054 | 0.859 |
| dense_k8 | 0.698 | 0.997 | 0.094 | 0.156 | 0.032 | 0.887 |
| dense_k16 | 0.531 | 0.998 | 0.062 | 0.125 | 0.038 | 0.849 |
| dense_k32 | 0.717 | 0.994 | 0.031 | 0.062 | 0.043 | 0.891 |
| rank_oracle_progress | 0.996 | 0.991 | 0.062 | 0.125 | 0.042 | 0.805 |
| rank_pairwise_oracle_action | -0.536 | 0.997 | 0.781 | 0.094 | 0.026 | 0.884 |
| rank_no_action | 0.487 | 0.997 | 0.188 | 0.219 | 0.019 | 0.979 |

## Interpretation

- The jobs are operationally healthy: training completed for all 18 variants,
  checkpoints exist, and all eval jobs are still running.
- The partial eval signal is negative on exact solving: `0/10` across all rows
  written so far, including oracle-goal symbolic re-encode rows.
- The ranking losses matter much more than dense horizon or hierarchy alone in
  the partial rows. Oracle-progress and oracle-action ranking are the only
  variants that get remaining Hamming below about 20.
- Predicted-goal planning is not competitive yet in the partial rows; best
  predicted-goal remaining Hamming is about 48, while best oracle-goal remaining
  Hamming is 14.5.
- We still cannot judge latent rollout or hierarchy-as-planner from this eval
  pass, because those rows have not been reached yet.
- However, failure in the symbolic re-encode rows is already enough to say this
  wave is not a faithful reproduction of the old Grid3 result. The old run used
  independent transition batches, local/context-weighted one-step and rollout
  MSE, no Grid-Token goal predictor or auxiliary geometry losses, and an
  overwrite-capable re-encoded/reset oracle planner.
