# Current Experiments

Last updated: 2026-07-02 17:16 CEST

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

Status at 15:42 CEST:

- `9/14` train jobs completed successfully; `5/14` are still running.
- `A_refactor_equiv_14816_dropout_off` failed from NaN latents during final
  diagnostics; its eval dependency is `DependencyNeverSatisfied`.
- `8` eval jobs are running and have started writing planner rows.
- Partial first-row signal: `A_refactor_equiv_14816`, `mpc_beam`, oracle raw
  L2, depth 4 solved `5/8`, h `0.375`. `A_smooth_14816_like` has an early
  `1/8` row. Current K16/K8 single-rollout and old-path H8/H16 first rows are
  `0/8`.

Status at 16:22 CEST:

- `13/14` train jobs completed successfully; the only failed train remains
  `A_refactor_equiv_14816_dropout_off` because of NaNs.
- `13` eval jobs are running; the dropout-off eval remains
  `DependencyNeverSatisfied`.
- Partial oracle results are much better than the earlier Clean17 wave:
  `A_anchor_repro` already has `8/8`, h `0.0` with `mpc_beam`, oracle raw L2,
  depth 16; `A_no_predict_delta` already has `8/8`, h `0.0` with `mpc_beam`,
  oracle raw L2, depth 4.
- Predicted-goal rows remain `0/8` so far, with best partial remaining Hamming
  around low 40s.

Follow-up dropout controls at 16:41 CEST:

- Code/config audit: `A_anchor_repro` and `A_refactor_equiv_14816` submitted
  configs differ only in `model.dense_rollout_refactor_mode`.
- Deterministic unit test now checks both loss and gradient equivalence between
  old and refactored objectives when dropout is `0.0`.
- Dropout locations: attention-weight dropout inside every
  `nn.MultiheadAttention`, plus two MLP `nn.Dropout` layers per transformer
  block. This affects context/state/predictor/goal/high-level/macro encoders,
  and the EMA target encoder while the model is in train mode.
- New control jobs submitted:

| Variant | Purpose | Train | Eval | State |
|---|---|---:|---:|---|
| `A_anchor_dropout_off_fp32` | old path, dropout off, fp32, LR `1e-4` | `3806051` | `3806052` | train running |
| `A_refactor_equiv_14816_dropout_off_fp32` | refactor, dropout off, fp32, LR `1e-4` | `3806053` | `3806054` | train running |
| `A_anchor_dropout_off_lr5e5` | old path, dropout off, bf16, LR `5e-5` | `3806055` | `3806056` | train running |
| `A_refactor_equiv_14816_dropout_off_lr5e5` | refactor, dropout off, bf16, LR `5e-5` | `3806057` | `3806058` | train running |

Follow-up at 17:01 CEST:

- The trainer now records `grad_norm_pre_clip`, records `grad_clip`, and
  raises immediately on non-finite loss or pre-clip gradient norm.
- The earlier dropout-off NaN jobs did use `training.grad_clip=1.0`, but
  pre-clip gradient norms were not logged, so we cannot tell whether large
  finite gradient norms preceded NaN.
- New lower-LR/fp32-batch4 controls submitted:

| Variant | Purpose | Train | Eval | State |
|---|---|---:|---:|---|
| `A_anchor_dropout_off_lr1e5` | old path, dropout off, bf16, LR `1e-5`, log every 100 steps | `3806110` | `3806111` | train running |
| `A_refactor_equiv_14816_dropout_off_lr1e5` | refactor, dropout off, bf16, LR `1e-5`, log every 100 steps | `3806112` | `3806113` | train running |
| `A_anchor_dropout_off_fp32_b4` | old path, dropout off, fp32, batch 4, grad accum 2 | `3806114` | `3806115` | train running |
| `A_refactor_equiv_14816_dropout_off_fp32_b4` | refactor, dropout off, fp32, batch 4, grad accum 2 | `3806116` | `3806117` | train running |

Status at 17:06 CEST:

- All four newly submitted dropout-off controls failed immediately with
  `Non-finite gradient norm at step 1: nan`; their eval jobs are
  `DependencyNeverSatisfied`.
- This confirms the earlier dropout-off runs did not merely diverge late from
  LR. They have a finite step-1 loss but non-finite gradients before clipping.
- The main non-dropout-off factorization evals have `96` planner rows so far
  and remain the useful result source.

Follow-up at 17:14 CEST:

- Initialization audit: the model mostly uses PyTorch defaults. Linear layers
  are already Kaiming-style defaults; embeddings are the high-scale part
  because each grid token sums seven `nn.Embedding` vectors initialized around
  unit standard deviation.
- Found and fixed a higher-priority NaN candidate: zero-weighted auxiliary
  objectives were still in the autograd graph as `0 * loss`. Disabled
  auxiliary losses are now gated out of computation and loss assembly.
- Submitted fresh gated dropout-off controls with `RUN_SUFFIX=_gated`:

| Variant | Train | Eval | State |
|---|---:|---:|---|
| `A_anchor_dropout_off_lr1e5_gated` | `3806182` | `3806183` | running |
| `A_refactor_equiv_14816_dropout_off_lr1e5_gated` | `3806184` | `3806185` | running |
| `A_anchor_dropout_off_fp32_b4_gated` | `3806186` | `3806187` | running |
| `A_refactor_equiv_14816_dropout_off_fp32_b4_gated` | `3806188` | `3806189` | running |

Initial health: all four gated controls passed the old step-1 failure point.
The bf16 LR `1e-5` anchor/refactor reached step `100` with finite pre-clip
grad norms around `58.7`; the fp32 batch-4 controls logged finite step-1 grad
norms around `151.9`.

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
