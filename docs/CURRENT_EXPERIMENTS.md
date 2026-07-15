# Current Experiments

Source of truth: `../sequence-editing-report/CURRENT_EXPERIMENTS.md`.

Last updated: 2026-07-15 10:47 CEST

## Joint End-to-End HWM Objective Gate

Wave 15 fixes exact N=2 valid rigid motion, one 256D MLP CLS, hierarchy
`[1,10,100]`, 8D ordered macro actions, causal Transformer predictors, and
four-step dense supervision at every level. Unlike Wave 14, every hierarchy
level and the shared encoder train jointly from step 0.

The 36 runs cross 12 online/EMA/SIGReg/VICReg/LDAD recipes with three seeds.
Trainer array `3858542` is active; correlated frozen-probe array `3858543` is
dependency-held. Output root:
`$HPCVAULT/sequence-editing/runs/controlled_objects/controlled_joint_hwm_objectives_v1_steps20000/`.

| stage | job | tasks | state at submission |
| --- | ---: | ---: | --- |
| joint training | `3858542` | 36 | 12 running; 24 array-limited |
| frozen probes | `3858543` | 36 | dependency-held per matching trainer |

GPU smoke `3858525` completed online, SIGReg, and LDAD paths `0:0` at batch
64. No production result is available yet. Planning is deliberately absent
from this gate: it will run with at least 32 shared episodes per seed and
confidence intervals only for representation-qualified objective cells.

HWM's shared latent, temporal scales, action chunks, first-latent subgoals, and
recursive receding-horizon control are present. Joint encoder optimization is
the requested end-to-end variant; the HWM paper's PLDM example freezes its
low-level encoder. Paper-style online SIGReg/LDAD rows are separated from
explicit EMA hybrids.
