# Current Experiments

Last updated: 2026-07-02 15:03 CEST

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
| `A_anchor_repro` | exact anchor | `3805527` | `3805528` | train running, eval dependency-held |
| `A_refactor_equiv_14816` | single shared rollout, old term structure | `3805529` | `3805530` | train running, eval dependency-held |
| `A_refactor_equiv_14816_dropout_off` | exact refactor, dropout `0.0` | `3805531` | `3805532` | train running, eval dependency-held |
| `A_smooth_14816_like` | count-weighted smooth approximation | `3805533` | `3805534` | train running, eval dependency-held |
| `A_uniform_k16` | one rollout to 16, uniform all-step weights | `3805535` | `3805536` | train running, eval dependency-held |
| `A_inv_sqrt_k16` | one rollout to 16, `1/sqrt(i)` weights | `3805537` | `3805538` | train running, eval dependency-held |
| `A_gamma_k16` | one rollout to 16, gamma `0.8^(i-1)` | `3805539` | `3805540` | train running, eval dependency-held |
| `A_inv_sqrt_k8` | one rollout to 8, `1/sqrt(i)` weights | `3805541` | `3805542` | train running, eval dependency-held |
| `A_old_path_h16_only` | old path, only horizon 16 | `3805543` | `3805544` | train running, eval dependency-held |
| `A_old_path_h8_only` | old path, only horizon 8 | `3805545` | `3805546` | train running, eval dependency-held |
| `A_no_goal_mse` | remove context-goal MSE | `3805547` | `3805548` | train running, eval dependency-held |
| `A_initial_current_goal` | `q(c,H0,Ht)` goal conditioning | `3805549` | `3805550` | train running, eval dependency-held |
| `A_no_hierarchy` | remove hierarchy training | `3805551` | `3805552` | train running, eval dependency-held |
| `A_no_predict_delta` | predict next latent directly | `3805553` | `3805554` | train running, eval dependency-held |

Status at 15:03 CEST:

- All 14 train jobs are still running on `rtxpro6k`; all eval jobs remain
  dependency-held.
- Latest train logs are between step `500` and `1500` out of `5000`.
- `13/14` train jobs have finite losses.
- `A_refactor_equiv_14816_dropout_off` is logging NaNs from step `500`; treat
  that diagnostic row as suspect unless it is rerun later. It has not been
  canceled.
- Based on current step rate and previous similar eval durations, first eval
  rows are expected around `16:00-16:30 CEST`; the full finite-result matrix is
  expected around `17:00-18:00 CEST` if the eval jobs start promptly after
  dependencies clear.

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
