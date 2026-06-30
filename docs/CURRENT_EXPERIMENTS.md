# Current Experiments

Last updated: 2026-06-30 12:19 CEST

Old-local-value fast wave is running.

Slurm:

- Train array `3797928`, tasks `0-17%18`, all running on `rtxpro6k`.
- Eval array `3797929`, tasks `0-17`, dependency-held on training.

It uses faithful old-style local value action conditioning
(`old_local_value`), full-board raw latent MSE scoring, `5000` train steps, LR
`1e-4`, EMA+VICReg, temporal straightening, predicted-goal progress
monotonicity, and `q(c,H0,Ht)` predicted goals. Eval compares latent rollout vs
symbolic re-encode, oracle vs predicted goals, beam depths `{1,4,16,32}`, and
hierarchy vs no hierarchy when applicable.

Smoke before final submission:

- Heavy variants `dense_k32` and `hier_l4_l16_hier_dense` completed a 2-step
  train smoke at batch 8/no accumulation after the dense all-step rollout fix.
