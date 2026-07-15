# Wave 15: Joint HWM Objective Gate

## Question

Does a jointly trained three-level HWM learn a usable single-CLS object state,
and which of online prediction, EMA, SIGReg, VICReg, and adjacent LDAD best
prevents representational collapse without destroying dynamics?

## Fixed Contract

- World: exact N=2 complete rigid objects on `16x16`; one valid atomic
  translation/rotation per primitive action.
- State: `Linear(768,256)+GELU`, one learned CLS latent; never a grid latent.
- Hierarchy: spans `[1,10,100]`, ordered action chunks compressed to 8D macro
  actions, separate causal Transformer predictors, four-step dense rollout at
  every level.
- Training: all levels, macro encoders, predictors, and the shared state
  encoder optimize jointly from step 0. There is no staged checkpoint or
  encoder freeze.
- HWM control: high-level first predicted latent is the lower-level subgoal;
  recursive receding-horizon planning remains implemented, but planning is
  deferred until a representation-qualified checkpoint exists.
- Evaluation: standardized frozen regression probes, categorical probes,
  frozen pixel decoder, rollout probes, rank/std, adjacent LDAD decoding, and
  conditional macro support/reachability diagnostics.

The shared-latent, temporal-scale, macro-action, subgoal, and receding-MPC
mechanics follow HWM. Joint encoder training is the requested end-to-end HWM
variant; the paper's PLDM instantiation instead freezes its low-level encoder.

## Sweep

Twelve objectives by seeds `{1707,2707,3707}` give 36 runs:

`online`, `ema`, `sigreg`, `ema_sigreg`, `vicreg`, `ema_vicreg`, `ldad`,
`ema_ldad`, `vicreg_ldad`, `ema_vicreg_ldad`, `sigreg_ldad`, and
`ema_sigreg_ldad`.

Online SIGReg and LDAD rows use the shared encoder with no EMA or target
stop-gradient. EMA combinations are explicitly hybrid controls. SIGReg uses
the LeJEPA convex weighting `.995 prediction + .005 SIGReg`; LDAD uses adjacent
categorical action decoding at weight `.1`; VICReg uses Wave 14's stable
variance/covariance pair `.05/17.866`.

## Status

Submitted 2026-07-15. Trainer array `3858542` has 36 tasks with concurrency
12. Correlated frozen-probe array `3858543` has 36 tasks and releases each task
only after its matching trainer succeeds. Output:
`$HPCVAULT/sequence-editing/runs/controlled_objects/controlled_joint_hwm_objectives_v1_steps20000/`.

GPU smoke array `3858525` ran online, SIGReg, and LDAD at batch 64; all three
completed `0:0` in 32-52 seconds. Full tests, focused controlled-object tests,
compileall, shell syntax, and diff checks pass.

## Gate

Promote only objective cells whose three seeds preserve useful rank and improve
semantic, spatial, foreground, and rollout readouts over matched initialization.
Only promoted cells receive fixed, adequately sampled planning evaluation with
confidence intervals. Capacity, macro-width, trajectory, and object-load axes
remain blocked.
