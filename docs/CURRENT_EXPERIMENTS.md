# Current Experiments

Last updated: 2026-06-30 12:00 CEST

No Grid Goal jobs are currently running. The H1-extra eval array elements were
canceled to free RTX Pro 6000 capacity for the proposed old-local-value fast
wave.

Old-local-value fast wave is implemented and ready to submit:

```bash
scripts/experiments/submit_grid_goal_oldlocal_fast.sh
```

It uses faithful old-style local value action conditioning
(`old_local_value`), full-board raw latent MSE scoring, `5000` train steps, LR
`1e-4`, EMA+VICReg, temporal straightening, predicted-goal progress
monotonicity, and `q(c,H0,Ht)` predicted goals. Eval compares latent rollout vs
symbolic re-encode, oracle vs predicted goals, beam depths `{1,4,16,32}`, and
hierarchy vs no hierarchy when applicable.
