# Experiment Plan

Source of truth: `../sequence-editing-report/BACKLOG.md` and
`../sequence-editing-report/CURRENT_EXPERIMENTS.md`.

Only Wave 14 is active. It fixes every factor except the two VICReg
regularization coefficients and seed:

- exact N=2 valid rigid-object motion on `16x16`;
- one `768 -> 256` MLP latent, never a grid latent;
- hierarchy `[1,10,100]` in valid environment-action time;
- separate causal Transformer predictors with four-step dense supervision;
- ordered nonlinear 8D macro actions;
- EMA `.99`, stop-gradient, and no LDAD/SIGReg;
- variance `{.05,1,10,29.409}` by adjusted covariance
  `{.1,1,10,17.866}` by three seeds.

The first gate is representation: latent rank/std, exact-N=2 shape, position,
area, relation, foreground reconstruction, and rollout transfer must improve
consistently over matched random initialization. Mixed object-count probes are
reported separately as OOD load generalization.

Only passing representation cells are interpreted for planning. Manual
low-level subgoal control must work; then retrieval, CEM, support CEM, and MPPI
test recursive HWM planning. Macro support/reachability metrics determine
whether continuous actions leave the learned action manifold.

Do not launch any backlog stage without the gate and an explicit decision.
