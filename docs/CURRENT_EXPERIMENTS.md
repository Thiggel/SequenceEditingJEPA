# Current Experiments

Last updated: 2026-07-03 11:18 CEST

Source of truth: `../sequence-editing-report/CURRENT_EXPERIMENTS.md`.

## Active: Delta-JEPA / Single-State Ablation

Implementation and fidelity fixes are complete. The first Delta submission used
one-step dynamics and was superseded after the horizon ablation showed
`K8_smooth_count` is the current best base. The old Delta evals were canceled;
replacement K8/smooth-count train variants were submitted as independent Slurm
jobs so each eval can depend on its matching checkpoint.

Scripts:

- `scripts/slurm/run_grid_goal_delta_jepa_train.slurm`
- `scripts/slurm/run_grid_goal_delta_jepa_eval.slurm`

Grid:

- full board latent with Delta-JEPA LDAD, crossing meaningful dynamics target
  modes and goal regularizer off/on: 6 variants
- single hidden-state board latent with learned CLS encoder token, causal
  history predictor, Delta-JEPA, and goal regularizer off/on: 2 variants

Delta-JEPA defaults in these jobs: `dynamics_target_mode=online_no_stopgrad`,
no SIGReg/VICReg, `delta_action_weight=10`, LDAD horizons `[1,2,3,4,5]`,
K8 dense rollout with `dense_rollout_all_steps=true` and
`dense_rollout_weighting=smooth_count`, and no temporal/ranking/corruption
auxiliaries.

Train/eval jobs:

| Variant | Train | Oracle eval | Predicted eval |
|---|---:|---:|---:|
| `FB_online_noema_nogoal` | `3808387` | `3808388` | n/a |
| `FB_online_noema_goal` | `3808389` | `3808390` | `3808391` |
| `FB_stopgrad_noema_nogoal` | `3808392` | `3808393` | n/a |
| `FB_stopgrad_noema_goal` | `3808394` | `3808395` | `3808396` |
| `FB_stopgrad_ema_nogoal` | `3808397` | `3808398` | n/a |
| `FB_stopgrad_ema_goal` | `3808399` | `3808400` | `3808401` |
| `SV_online_nogoal` | `3808402` | `3808403` | n/a |
| `SV_online_goal` | `3808404` | `3808405` | `3808406` |

Eval is dependency-held per checkpoint and split by goal-distance mode:

- oracle rows use `oracle_goal_raw_euclidean_distance`
- predicted rows use `predicted_goal_raw_euclidean_distance` only for
  goal-trained variants
- output dirs are suffixed with `_oracle` or `_predicted`
- each eval uses `mpc_beam`, latent rollout plus symbolic re-encode, beam width
  `16`, depths `{4,16}`, and 8 boards

Current Slurm state at 11:18 CEST:

- replacement train jobs `3808387`, `3808389`, `3808392`, `3808394`,
  `3808397`, `3808399`, `3808402`, `3808404` are pending on
  `rtxpro6k,a100`
- replacement eval jobs are dependency-held
- superseded one-step Delta evals `3808222`, `3808224`, `3808225`, `3808227`,
  `3808229`, `3808230`, `3808232`, `3808234`, `3808235`, `3808237`,
  `3808239`, `3808240` were canceled

Fidelity fixes:

- `delta_action_weight > 0` now requires non-empty `delta_action_horizons`.
- `use_ema_target_encoder=true` is rejected with `online_no_stopgrad`
  dynamics, because that combination would be a no-op.
- Single-state latent-rollout planning now passes growing state/action history
  to the causal predictor.
- Full-board Delta goal variants use `goal_conditioning=context_current`.

## Active: Horizon-Length Ablation

This sweep tests whether the multi-step dynamics horizon itself is the
important factor, using only the clean one-long-rollout path. It does not use
the old legacy multi-horizon rollout code.

Fixed base:

- dropout off: `model.dropout=0.0`
- no residual delta: `model.predict_delta=false`
- one recursive rollout: `model.dense_rollout_all_steps=true`
- no hierarchy: `model.hierarchy_levels=[]`
- context-goal MSE on, goal NCE off
- no SIGReg/VICReg, temporal straightening, progress rank, action rank, or
  terminal corruption
- seed `5204`, LR `1e-4`, batch `8`, `5000` steps

Grid:

| Horizon | Uniform job | Smooth/count job |
|---:|---|---|
| 1 | train `3807867`, eval `3807868` | train `3807869`, eval `3807870` |
| 2 | train `3807871`, eval `3807872` | train `3807873`, eval `3807874` |
| 3 | train `3807875`, eval `3807876` | train `3807877`, eval `3807878` |
| 4 | train `3807879`, eval `3807880` | train `3807881`, eval `3807882` |
| 8 | train `3807883`, eval `3807884` | train `3807885`, eval `3807886` |
| 16 | train `3807887`, eval `3807888` | train `3807889`, eval `3807890` |

Eval is flat latent-rollout MPC beam only: beam width `16`, depths `{4,16}`,
8 boards, oracle raw L2 and predicted raw L2.

Current state at 11:00 CEST:

- K1-K4 train/eval pairs completed successfully; eval rows are available.
- K8/K16 training completed successfully, but eval jobs `3807884`, `3807886`,
  `3807888`, and `3807890` failed at checkpoint load after the Delta-JEPA
  decoder was added. The loader instantiated a Delta decoder for these older
  horizon checkpoints and reported missing `delta_action_decoder.*` keys.
- The loader compatibility fix is applied. Repair evals are running:
  `K8_uniform` `3808345`, `K8_smooth_count` `3808346`, `K16_uniform`
  `3808347`, `K16_smooth_count` `3808348`.
- First repair rows: `K8_smooth_count` solves `8/8` with oracle raw L2 at
  depth 4; `K8_uniform` reaches h `3.75`; `K16_smooth_count` reaches h
  `4.375`; predicted-goal repair rows are still pending.

## Previous Sweep Takeaway

The completed dropout-off factorization sweep showed that oracle-goal planning
is recoverable, but predicted-goal planning remains `0/8` across all rows.
Dropout-off rescued exact-refactor, smooth/count, and old H8-only objectives,
but did not rescue uniform/gamma/K16 single-rollout losses.
