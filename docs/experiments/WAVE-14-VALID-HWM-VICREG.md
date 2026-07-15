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

Complete. All 48 tasks in each of train arrays `3855790` (`[1]`), `3855791`
(`[1,10]`), and `3855792` (`[1,10,100]`) and probe/planner array `3855793`
completed with exit `0:0`. The output contains 48/48 final probes and no
missing cell.

## Results

| representative variance / covariance | rank `/256` | presence BA | shape BA | position R2 | foreground IoU | prediction / rollout MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `.05 / .1` | 15.9 | .699 | .293 | .238 | .063 | .061 / .079 |
| `.05 / 17.866` | 60.1 | .751 | .275 | -.211 | .166 | .017 / .021 |
| `1 / 17.866` | 94.3 | .665 | .235 | -2.170 | .146 | .297 / .344 |
| `29.409 / 17.866` | 61.7 | .610 | .252 | -1.788 | .098 | .330 / .386 |

Covariance weight monotonically raises rank within each variance row, but
higher variance weights sharply damage prediction and semantic readout. The
best dynamics/semantic compromise is `.05 / 17.866`: all seeds have rank
`59.7-60.7`, presence BA `.748-.757`, positive shape gain, and prediction MSE
`.015-.021`. It still loses roughly 71 effective dimensions from initialization,
has negative position/relation R2, and lowers foreground IoU from initialization
in every seed. The highest-rank row (`1 / 17.866`, rank 94.3) has much worse
prediction, position, shape, and reconstruction. Mixed-load object-count BA is
`.684` for the compromise row versus `.599` at matched initialization.

Action-delta probes in the compromise row decode row `.250`, column `.187`,
transform `.476`, and selected color `.170`; predicted-delta transfer retains
row `.236`, column `.175`, transform `.590`, and color `.149`. The latent
therefore carries action effects more clearly than object geometry.

Direct level-1/2 endpoint MSE is `.018/.024`, but realizing those endpoints by
primitive rollout gives `.139/5.491`. Reachability AUROC is `.680/.509` and
support AUROC `.184/.188`. The support diagnostic and support-CEM penalty use a
joint 256D state plus 8D macro nearest-neighbor distance; state distance
dominates and the measured support score is not a valid off-manifold detector.

Two evaluation caveats prevent stronger claims. Area R2 uses tiny unstandardized
targets and produces nonsensical large negative values, so it is excluded.
Planning used one episode per seed, so success rates have only three trials per
coefficient pair. Staged training also freezes the shared encoder after `[1]`;
this wave tests low-level VICReg representation rescue, not whether higher-level
losses induce abstraction.

## Conclusion

The representation gate fails. VICReg prevents total constant collapse and
exposes a reproducible rank/dynamics tradeoff, but no coefficient pair jointly
preserves rank and improves semantic and foreground reconstruction in all
three seeds. Planning numbers are not promoted. No further training wave is
active; the next decision should first repair/recalibrate the frozen probes and
conditional macro-support diagnostic on retained checkpoints.
