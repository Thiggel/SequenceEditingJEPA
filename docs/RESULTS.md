# Results

Long-form results: `../sequence-editing-report/RESULTS.md`.

Last updated: 2026-07-15 10:47 CEST

Wave 15 joint HWM trainers `3858542` and dependent probes `3858543` are active;
there are no production results yet. GPU smoke `3858525` completed the online,
SIGReg, and LDAD paths at batch 64 with exit `0:0`.

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
