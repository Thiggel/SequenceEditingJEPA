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
- Status: complete 231/231 trainers and probes, all exit `0:0`

## Gate

Compare all frozen property probes, pixel-decoder reconstruction, latent rank,
transition/rollout losses, and hierarchy diagnostics over all three seeds. A
low prediction MSE is not success when rank or frozen-feature utility collapses.

## Results

| recipe | rank | shape BA | position R2 | relation R2 | foreground IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| stop-gradient only | 2.5 | .339 | .472 | .278 | .019 |
| EMA only | 3.2 | .313 | .469 | .258 | .019 |
| VICReg stopgrad 1x | 50.3 | .291 | .550 | .165 | .153 |
| VICReg EMA 1x | 47.6 | .280 | .531 | .086 | .155 |
| VICReg stopgrad 10x | 125.6 | .236 | -.294 | -.187 | .248 |
| VICReg stopgrad 100x | 162.3 | .226 | -1.210 | -.371 | .211 |
| SIGReg stopgrad 1x | 17.8 | .279 | .346 | -4.197 | .052 |
| LDAD online 1x | 9.4 | .237 | .293 | -.932 | .064 |

VICReg 1x is the best useful spatial/semantic representation but loses about
81 effective dimensions and `.058` foreground IoU from initialization. VICReg
10x nearly preserves rank and improves foreground in every seed, but has
negative absolute position/relation R2 and inconsistent shape gain. VICReg
100x raises rank above initialization without restoring semantics. LDAD,
SIGReg, and their combinations do not remove this tradeoff. The gate fails.
