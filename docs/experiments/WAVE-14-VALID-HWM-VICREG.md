# Wave 14: Valid-Motion HWM VICReg

## Question

With one fixed, valid, paper-shaped hierarchy, can EMA plus appropriately
weighted VICReg prevent collapse and yield useful object representations?

## Fixed Contract

- State: one MLP latent, `Linear(768,256)+GELU`; no grid latent.
- World: exactly two complete rigid objects on `16x16`, random shapes/colors,
  hidden motion policies independent of color.
- Primitive action: `(row,column,transform)` selects an object pixel and applies
  one translation or rotation atomically. A frame changes 2-10 sparse cells at
  once; every observed state contains complete objects.
- Levels: `[1]`, `[1,10]`, then `[1,10,100]`; lower levels and the shared
  encoder are frozen when the next level is trained.
- Predictor: two-layer causal Transformer, four heads, four deeply supervised
  teacher-forced and autonomous MSE rollout steps at every level.
- Macro action: ordered nonlinear action-chunk encoder to an 8D bottleneck.
- Target: EMA `.99` with stop-gradient.
- Planning: recursive first-subgoal HWM; on-support retrieval, 10-iteration
  CEM with `.7` variance EMA, 10-iteration MPPI, support energy,
  symbolic/replay controls, and a ground-truth low-level subgoal diagnostic.

## Sweep

VICReg standard-deviation coefficient `{.05,1,10,29.409}` by adjusted
off-diagonal covariance coefficient `{.1,1,10,17.866}` by seeds
`{1707,2707,3707}`: 48 final cells, 144 staged trainers, 48 final probe/planner
evaluations. All other axes are fixed.

## Gates

Representation comes first: effective rank, per-dimension std, shape,
position, relation, area, foreground reconstruction, and all rollout probes.
These are reported on exact N=2 scenes; a separate mixed N=1/2/4/8 suite
measures object-count and load generalization without contaminating the primary
in-distribution readout.
The randomly assigned hidden motion policy is unobservable from one frame and
is reported only as a negative control, not as a representation success gate.
Planning is interpretable only if the manual low-level subgoal control works.
Macro support/reachability diagnostics determine whether continuous macro
actions produce off-manifold or low-level-unreachable subgoals.

## Status

Submitted 2026-07-14 after the full repository test suite and three-stage CUDA
smoke `3855783` passed. Train arrays are `3855790` (`[1]`), `3855791`
(`[1,10]`), and `3855792` (`[1,10,100]`); probe/planner array `3855793`
depends on the final stage. Sixteen low-level tasks are currently running.

## Results

No trained final cell is complete at this snapshot. The CUDA smoke verifies
execution only: valid-action and object-area-preservation fractions are `1.0`,
and sampled actions change `2-6` cells simultaneously.

## Conclusion

Pending all three seeds for every coefficient pair. Do not interpret hierarchy
or planner differences until the representation and manual-subgoal gates are
evaluated.
