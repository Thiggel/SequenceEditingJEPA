# Current Experiments

Last updated: 2026-07-03 09:22 CEST

Source of truth: `../sequence-editing-report/CURRENT_EXPERIMENTS.md`.

## Minimal-Aux Factorization Final Readout

All current minimal-aux factorization and dropout-off loss-scheme jobs have
finished, except stale superseded pre-fix eval dependencies. The sweep produced
`252` planner rows across `32` evaluated variants.

Main result:

- Oracle-goal latent planning works for the original anchor, no-delta,
  dropout-off exact-refactor fp32 batch-4, dropout-off smooth/count weighting,
  and dropout-off old H8-only.
- Predicted-goal planning remains unsolved: every predicted-goal row is `0/8`.
- Dropout was a major hidden variable for the refactored/smooth objectives,
  but uniform/gamma/K16 single-rollout objectives are still structurally weaker.
- Operational default changed after this result: `model.dropout` is now `0.0`
  in `configs/puzzle/grid_goal_sudoku.yaml`.

Best rows:

| Variant | Intuition | Best oracle | Best predicted |
|---|---|---|---|
| `A_anchor_repro` | original minimal-aux recipe | `8/8`, h `0.0` | `0/8`, h `44.2` |
| `A_no_predict_delta` | predict next latent directly | `8/8`, h `0.0` | `0/8`, h `41.4` |
| `A_refactor_equiv_14816` | refactored old loss, dropout on | `5/8`, h `0.375` | `0/8`, h `40.4` |
| `A_refactor_equiv_14816_dropout_off_fp32_b4_gated` | refactored old loss, dropout off, fp32 batch-4 | `8/8`, h `0.0` | `0/8`, h `38.2` |
| `A_smooth_14816_like_dropout_off_bf16_l1e4` | smooth/count loss, dropout off | `8/8`, h `0.0` | `0/8`, h `33.5` |
| `A_smooth_14816_like_dropout_off_fp32_b4` | smooth/count loss, dropout off, fp32 batch-4 | `8/8`, h `0.0` | `0/8`, h `34.9` |
| `A_old_path_h8_only_dropout_off_bf16_l1e4` | old separate rollout, H8 only, dropout off | `6/8`, h `0.25` | `0/8`, h `31.8` |
| `A_old_path_h8_only_dropout_off_fp32_b4` | old separate rollout, H8 only, dropout off, fp32 batch-4 | `8/8`, h `0.0` | `0/8`, h `32.9` |
| `A_uniform_k16_dropout_off_fp32_b4` | one K16 rollout, uniform weights, dropout off | `0/8`, h `7.5` | `0/8`, h `40.1` |
| `A_inv_sqrt_k16_dropout_off_bf16_l1e4` | one K16 rollout, inverse-sqrt weights, dropout off | `0/8`, h `19.6` | `0/8`, h `42.4` |
| `A_gamma_k16_dropout_off_fp32_b4` | one K16 rollout, geometric weights, dropout off | `0/8`, h `30.8` | `0/8`, h `45.0` |

Interpretation:

- Removing dropout rescues exact-refactor, smooth/count, and old H8-only.
- Removing dropout does not rescue uniform/gamma/K16 enough to solve.
- H16-only remains bad with or without dropout.
- Removing residual delta prediction did not hurt in this setup.
- Context-only goal MSE matters for oracle geometry; removing it drops to
  `0/8`.
- The next bottleneck is predicted-goal geometry, not oracle-goal planning.
