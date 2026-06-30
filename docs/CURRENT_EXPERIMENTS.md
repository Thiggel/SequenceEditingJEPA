# Current Experiments

Last updated: 2026-06-30 11:42 CEST

No Grid Goal jobs are currently running. The H1-extra eval array elements were
canceled to free RTX Pro 6000 capacity for the proposed old-local-value fast
wave.

Next proposed sweep: faithful old-style local value action conditioning with
full-board raw latent MSE scoring, `5000` train steps, LR `1e-4`, EMA+VICReg,
temporal straightening, progress monotonicity, and goal prediction. The sweep
should compare latent rollout vs symbolic re-encode, oracle vs predicted goals,
beam depths `{1,4,16,32}`, and hierarchy vs no hierarchy when applicable.
