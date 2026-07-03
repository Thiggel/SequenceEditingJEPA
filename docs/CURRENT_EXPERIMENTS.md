# Current Experiments

Last updated: 2026-07-03 09:45 CEST

Source of truth: `../sequence-editing-report/CURRENT_EXPERIMENTS.md`.

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

Initial Slurm state: all 12 train jobs started immediately on `rtxpro6k`
nodes `a2143` and `a2041`; all 12 eval jobs are dependency-held.

## Previous Sweep Takeaway

The completed dropout-off factorization sweep showed that oracle-goal planning
is recoverable, but predicted-goal planning remains `0/8` across all rows.
Dropout-off rescued exact-refactor, smooth/count, and old H8-only objectives,
but did not rescue uniform/gamma/K16 single-rollout losses.
