# Current Experiments

Source of truth: `../sequence-editing-report/CURRENT_EXPERIMENTS.md`.

Last updated: 2026-07-15 09:58 CEST

## Fixed Valid-Motion HWM VICReg Sweep

Wave 14 is complete and no new experiment is active. It tested whether EMA
plus a VICReg variance/covariance coefficient pair could give one fixed
single-CLS `[1,10,100]` rigid-motion HWM a usable state representation. All
192 jobs completed `0:0`: trainers `3855790`-`3855792` and final evaluations
`3855793`, with 48/48 three-stage cells and 48/48 probe files.

| representative variance / covariance | rank `/256` | presence BA | shape BA | position R2 | foreground IoU | prediction / rollout MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `.05 / .1` | 15.9 | .699 | .293 | .238 | .063 | .061 / .079 |
| `.05 / 17.866` | 60.1 | .751 | .275 | -.211 | .166 | .017 / .021 |
| `1 / 17.866` | 94.3 | .665 | .235 | -2.170 | .146 | .297 / .344 |
| `29.409 / 17.866` | 61.7 | .610 | .252 | -1.788 | .098 | .330 / .386 |

No coefficient pair passes the preregistered representation gate. Covariance
pressure raises rank, but its high-rank cells lose dynamics and semantic
quality. The low-variance `.05` row is the only stable useful region for
presence, shape, action information, and prediction, but its rank still falls
from about 131 at initialization to 60 and its best foreground IoU `.166` is
below matched initialization by `.039`. Position and relation R2 remain
negative in that row. Area R2 is excluded because the unstandardized tiny-area
targets make the 200-step gradient probe numerically invalid.

At `.05 / 17.866`, direct 10/100-action endpoint MSE is `.018/.024`, while
primitive realization MSE is `.139/5.491`. Level-1 reachability AUROC is `.680`,
but level-2 is `.509`; the support AUROCs are `.184/.188`. The current joint
state/macro nearest-neighbor support score is confounded by its 256D state
distance and should be repaired before treating support-CEM as an off-manifold
solution. Planner success rates are descriptive only because this wave used
one episode per seed.

The sweep also cannot establish hierarchy-induced abstraction: staged training
freezes the encoder after `[1]`, so the 10- and 100-action losses cannot change
the representation. No follow-up training has been submitted.

Results and retained artifacts:
`$HPCVAULT/sequence-editing/runs/controlled_objects/controlled_valid_hwm_vicreg_v1_steps20000/`.
The aggregate is `summary.json`. Forty-eight final `[1,10,100]` checkpoints,
all metrics/probes/manifests, and Slurm logs are retained; 96 redundant
intermediate checkpoints were removed, reducing the run root from 9.5G to
4.0G.
