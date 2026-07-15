# Wave 16: Objective-Weight Calibration

## Question

Can LDAD, VICReg, SIGReg, target stop-gradient, or EMA preserve a useful
single-CLS object representation when their strengths are calibrated rather
than compared at one arbitrary coefficient?

## Fixed Contract

- Exact two-object valid rigid-motion trajectories.
- One learned 256D MLP CLS; no grid latent.
- Jointly trained hierarchy `[1,10,100]`, causal Transformer predictors, 8D
  nonlinear ordered action chunks, and four-step rollout supervision.
- 20,000 optimizer steps and seeds `1707,2707,3707`.
- Coefficient multipliers `{1,10,100}` around LDAD base `.1`, VICReg outer
  base `1` with variance/covariance `.05/17.866`, and SIGReg base `.005`.

## Grid

The 77 recipes comprise stop-gradient and EMA controls; LDAD under online,
stop-gradient, and EMA targets; standalone VICReg/SIGReg under stop-gradient
and EMA; and every LDAD x VICReg or LDAD x SIGReg multiplier pair under
online, stop-gradient, and EMA contracts. VISReg is excluded. Three seeds give
231 trainers and 231 correlated frozen probes.

## Execution

- Trainers: `3860384` (`0-230%24`)
- Correlated probes: `3860385`, dependency `aftercorr:3860384`
- Root: `$HPCVAULT/sequence-editing/runs/controlled_objects/controlled_objective_weights_v1_steps20000`
- Status: active, submitted 2026-07-15

## Gate

Compare all frozen property probes, pixel-decoder reconstruction, latent rank,
transition/rollout losses, and hierarchy diagnostics over all three seeds. A
low prediction MSE is not success when rank or frozen-feature utility collapses.

