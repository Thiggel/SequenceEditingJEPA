# Results

Long-form results: `../sequence-editing-report/RESULTS.md`.

Last updated: 2026-07-15 18:04 CEST

Waves 16-18 are active under jobs `3860384/3860385`, `3860420/3860421`, and
`3860422`. They test calibrated no-VISReg objectives, whole-trajectory dense
hierarchy training, and conditional-support/lower-reachability planning. No
production aggregate is complete; partial array means are not reported.

Wave 15 joint HWM trainers `3858542` and dependent probes `3858543` completed
36/36 each with exit `0:0`. VICReg is the strongest semantic/spatial row: rank
`33.3`, presence `.807`, shape `.304`, position `.705`, relation `.511`, and
foreground IoU `.116`. EMA+VICReg retains more rank (`47.6`) and foreground
`.155`, but weaker semantics. Initialization rank is about 131 and foreground
IoU about `.211`, so no objective passes the strict representation gate.

VICReg direct span-10/100 MSE is `.0020/.0030`, while primitive realization is
`.164/13.315`. Conditional macro support AUROC is repaired at about `.998`, but
level-2 reachability is chance at `.515`. No planning jobs were submitted.

Wave 14 completed all 192 jobs `3855790`-`3855793` with exit `0:0` and
48/48 final evaluations. No VICReg pair passes the representation gate.
Covariance pressure raises effective rank as high as `94.3/256`, but that row
has poor prediction and semantics. The best compromise, variance `.05` and
covariance `17.866`, has rank `60.1`, presence BA `.751`, shape BA `.275`,
position R2 `-.211`, relation R2 `-.979`, foreground IoU `.166`, and
prediction/rollout MSE `.017/.021`. Foreground reconstruction is worse than
matched initialization in every seed.

For that row, direct 10/100-action endpoint MSE is `.018/.024`, versus
primitive realization MSE `.139/5.491`. Level-2 reachability is chance and the
joint state/macro support score is invalid as an off-manifold detector. Planner
success used one episode per seed and is not promoted. Area R2 is also excluded
because its unstandardized tiny targets make the gradient probe numerically
invalid.

The staged encoder is frozen after `[1]`; Wave 14 therefore evaluates
low-level VICReg representation rescue and higher-level prediction/planning,
but not hierarchy-induced changes to the representation. Per-wave history is
indexed in `docs/experiments/README.md`; staged decisions are in
`docs/BACKLOG.md`.
