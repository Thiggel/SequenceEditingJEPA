# Current Experiments

Last updated: 2026-07-02 14:50 CEST

Source of truth: `../sequence-editing-report/CURRENT_EXPERIMENTS.md`.

## Active: Minimal-Aux Objective Factorization

This sweep isolates why the successful `minimal_aux` `[1,4,8,16]` objective
worked while later Clean17-style dense objectives failed. The key hypothesis is
that the old recipe combined context-goal MSE, residual delta prediction,
hierarchy training, and multi-resolution endpoint anchoring, while the failed
Clean17 runs changed multiple factors at once.

Common anchor:

- seed `5204`, LR `1e-4`, batch `8`, no grad accumulation, `5000` steps
- `action_conditioning=affected_marker`
- `predict_delta=true`
- EMA target encoder, no SIGReg/VICReg
- `goal_conditioning=context`, `goal_mse_weight=1.0`, no goal NCE
- `dense_future_weight=1.0`
- old non-variable dense path: `multi_step_horizons=[1,4,8,16]`
- hierarchy `[4,16]`, hierarchy loss `1.0`
- no temporal straightening, progress rank, action rank, or terminal corrupt

Old dynamic objective:

```text
L_old =
  E_1^direct
  + sum_{H in {4,8,16}} w_H E_{H,H}
  + (1 / 28) sum_{H in {4,8,16}} sum_{i=1..H} w_i E_{H,i}
```

where `w_i=1/sqrt(i)`, `E_1^direct` is the direct adjacent transition loss,
`E_{H,H}` is the terminal endpoint loss for rollout horizon `H`, and
`E_{H,i}` is the old per-term intermediate i-step loss inside the separate
rollout to horizon `H`.

Planned train/eval jobs:

| Variant | Change from anchor | Train | Eval | State |
|---|---|---:|---:|---|
| `A_anchor_repro` | exact anchor | pending | pending | not submitted yet |
| `A_refactor_equiv_14816` | single shared rollout, old term structure | pending | pending | not submitted yet |
| `A_refactor_equiv_14816_dropout_off` | exact refactor, dropout `0.0` | pending | pending | not submitted yet |
| `A_smooth_14816_like` | count-weighted smooth approximation | pending | pending | not submitted yet |
| `A_uniform_k16` | one rollout to 16, uniform all-step weights | pending | pending | not submitted yet |
| `A_inv_sqrt_k16` | one rollout to 16, `1/sqrt(i)` weights | pending | pending | not submitted yet |
| `A_gamma_k16` | one rollout to 16, gamma `0.8^(i-1)` | pending | pending | not submitted yet |
| `A_inv_sqrt_k8` | one rollout to 8, `1/sqrt(i)` weights | pending | pending | not submitted yet |
| `A_old_path_h16_only` | old path, only horizon 16 | pending | pending | not submitted yet |
| `A_old_path_h8_only` | old path, only horizon 8 | pending | pending | not submitted yet |
| `A_no_goal_mse` | remove context-goal MSE | pending | pending | not submitted yet |
| `A_initial_current_goal` | `q(c,H0,Ht)` goal conditioning | pending | pending | not submitted yet |
| `A_no_hierarchy` | remove hierarchy training | pending | pending | not submitted yet |
| `A_no_predict_delta` | predict next latent directly | pending | pending | not submitted yet |

Eval per checkpoint is an independent dependency-held job:

- transition: `latent_rollout`
- planner: `mpc_beam`, plus `hierarchical_beam` when hierarchy exists
- beam width `16`, depths `{4,16}`
- `8` boards
- scores: `oracle_goal_raw_euclidean_distance` and
  `predicted_goal_raw_euclidean_distance`
- diagnostics skipped for speed

Implementation:

- `model.dense_rollout_refactor_mode=legacy_equivalent` reproduces the old
  dense endpoint/intermediate term structure from one shared rollout.
- `model.dense_rollout_refactor_mode=legacy_count` uses the old horizon-count
  shape without the extra endpoint terms.
- Regression tests cover exact refactor equivalence with dropout off,
  count-weight semantics, incompatible dense-mode validation, and all Slurm
  variant/eval arguments.

Verification before submission:

- `source scripts/env.sh && pytest -q tests/test_grid_goal_jepa.py`
- `source scripts/env.sh && pytest -q tests/test_grid_goal_next_wave_scripts.py`
- `source scripts/env.sh && python -m compileall -q puzzle_jepa scripts tests`
