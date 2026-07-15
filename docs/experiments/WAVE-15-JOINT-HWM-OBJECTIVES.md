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

Complete. Trainer array `3858542` and correlated frozen-probe array `3858543`
both completed 36/36 tasks with exit `0:0`. Output:
`$HPCVAULT/sequence-editing/runs/controlled_objects/controlled_joint_hwm_objectives_v1_steps20000/`.

GPU smoke array `3858525` ran online, SIGReg, and LDAD at batch 64; all three
completed `0:0` in 32-52 seconds. Full tests, focused controlled-object tests,
compileall, shell syntax, and diff checks pass.

## Results

| objective | rank `/256` | presence BA | shape BA | position R2 | relation R2 | foreground IoU | prediction / rollout MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| online | 1.3 | .777 | .314 | .030 | -.124 | .012 | .0000002 / .0000002 |
| EMA | 3.2 | .793 | .313 | .468 | .230 | .020 | .0000002 / .0000003 |
| SIGReg | 9.2 | .657 | .261 | .439 | -2.627 | .042 | .040 / .052 |
| LDAD | 9.4 | .653 | .241 | .292 | -.925 | .063 | .0045 / .0052 |
| VICReg | 33.3 | .807 | .304 | .705 | .511 | .116 | .0021 / .0028 |
| EMA+VICReg | 47.6 | .782 | .279 | .531 | .086 | .155 | .0079 / .0100 |

VICReg is the best semantic/spatial cell and EMA+VICReg retains the most rank
and foreground information. VICReg position R2 stays positive through four
rollouts at spans 1/10/100 (`.698/.644/.579`), and predicted deltas decode the
transform at `.708`. These gains reproduce across all three seeds.

Matched initialization has effective rank about 131 and foreground IoU about
`.211`. Every objective loses rank and every foreground decoder gets worse.
Online/EMA prediction collapses almost to a constant state. Direct VICReg
10/100-action MSE is `.0020/.0030`, but primitive realization MSE is
`.164/13.315`; level-2 reachability AUROC is `.515`. Conditional support AUROC
is now `.998`, showing that support detection is repaired but does not imply
that predicted subgoals are reachable.

## Gate

The strict gate fails: no objective both preserves representation breadth and
improves semantic, spatial, foreground, and rollout readouts. No planning jobs
are submitted. VICReg and EMA+VICReg are retained as diagnostic checkpoints;
the next experiment requires an explicit decision rather than another broad
sweep.
