# Current Experiments

Source of truth: `../sequence-editing-report/CURRENT_EXPERIMENTS.md`.

Last updated: 2026-07-10 16:15 CEST

## Object Dynamics JEPA Scaffold

Status: fidelity-audited and repaired; prestage is ready but not yet submitted.

Purpose: test whether LeWM-like compressed single-CLS JEPA dynamics can learn
hidden object/process structure from low-level grid edits. The model is not
given object slots, object IDs, or proposal IDs during training. Hidden object
metadata is used only to generate trajectories and run frozen probes.

Trajectory configs:

| Config | Meaning |
|---|---|
| `object_blocked` | T1: one hidden object is completed before the next. |
| `frontier_build` | T2: objects grow through 8-neighbor frontiers. |
| `random_within_object` | T3: object identity is blocked, local growth is removed. |
| `interleaved_build` | T4: persistent object processes are interleaved. |
| `global_random` | T5: final target has objects but edit order weakens temporal object signal. |
| `noisy_repair` | T6: objects are damaged/overgrown/recolored and repaired. |
| `completion` | Non-empty partial objects are completed. |
| `transform_identity` | Objects are transformed/recolored while identity is preserved. |
| `random_off_manifold` | Pure random-board/random-edit negative control. |

Prepared grids:

| Grid | Jobs | Submitted |
|---|---:|---|
| Prestage LR/steps | 12 dry-run commands | no |
| Phase trajectory/model/objective sweep | 104 dry-run commands | no |

Prepared scripts:

- `scripts/slurm/run_object_dynamics_train.slurm`
- `scripts/experiments/submit_object_dynamics_prestage.sh`
- `scripts/experiments/submit_object_dynamics_phase1.sh`

Verification:

- Twenty targeted object/Grid-Goal fidelity checks pass; nine remaining object
  research gaps are strict xfails in
  `tests/test_object_dynamics_remaining_fidelity.py`.
- The complete repository run is `297 passed, 9 xfailed` (`306` collected).
- Slurm verification `3830903` completed `0:0` on `a0123` in 20s; its log is
  `logs/jepa-obj-verify-3830903.out`. Preflight `3830803` failed `127:0`
  before collection because the repo-local interpreter was unavailable on the
  compute node; `logs/jepa-audit-verify-3830803.err` records the failure.
- One-step Hydra CPU runs pass for base, LDAD, VICReg, SIGReg, EMA, and H16
  hierarchy/noisy-repair configurations.
- LDAD now decodes encoded adjacent-state displacement with a shared
  end-to-end encoder; SIGReg now uses projected Epps-Pulley Gaussian testing.
- Effective semantic/counterfactual/wrong sampling is tested at `80/15/5`.
  Counterfactuals are local wrong-color/outgrowth/erase alternatives rather
  than shuffled gold actions; hidden per-state ownership keeps probes aligned
  to only the objects visible at each state.
- Frozen evaluation now includes visible object geometry/color/shape/relations,
  missing/overgrowth/wrong-color severity, balanced foreground grid/object-map
  decoding, latent-delta actions, rollout transfer, hierarchy chunks, latent
  rank/nearest neighbors, geometry-based off-manifold surprise, and matched
  raw-grid baselines on a fixed held-out set plus a step-0 baseline.
- Grid-Goal SIGReg now uses the same projected Epps-Pulley test, and Grid-Goal
  LDAD now decodes encoded endpoint displacement. Historical checkpoints were
  trained before these repairs and retain the old objective semantics.

Phase submission remains gated on a Phase-1 full-grid compression baseline,
paired full-grid/single-CLS Delta rows, long-horizon sequence LDAD, multi-seed
launch support, actual HWM planning, and explicit part/inside, attention,
nonlinear-probe, correction-chunk, and reconstruction baselines. The base-only
prestage does not violate those gates.

## Structured JEPA Audit Update

The 46-job structured wave is no longer running. Eighteen variants produced
`144` planner rows. `S0_cell_baseline` and full-grid `DJ0`-`DJ3` each solve
`8/8` with oracle-goal latent rollout; all evaluated single-CLS Delta and
combination rows remain `0/8` near random. Diagnostics expose a key caveat:
Historical Grid-Goal LDAD trained on predictor-produced displacement, so
perfect predicted-delta decoding can coexist with near-random encoded-target
delta decoding. The training path is repaired for future checkpoints only.

Fourteen final step-5000 checkpoints had zero planner rows because structured
slot latents had 82/108/110 tokens but the planner passed an 81-cell mask. The
mask expansion is fixed and a 12-hour dry-run repair launcher is available at
`scripts/experiments/submit_grid_goal_structured_eval_repair.sh`.
The repair template admits `a40,rtxpro6k,a100`; A40 was the freest suitable
partition at the pre-submission check.

## ARC First-Pass Candidate Scorers

Status: implemented, submitted, rerun with explicit context active masks, and
complete.

Purpose: after making ARC state/action sampling concrete, run the first three
actual train jobs on ARC candidate scoring:

- `raw_grid_energy`: context/query/candidate grid energy;
- `proposal_energy`: raw grid energy plus symbolic action/proposal features;
- `jepa_energy`: proposal-aware energy plus JEPA-style successor latent
  prediction from `(current state, action)`.

Jobs:

| Variant | Job | State | Output |
|---|---:|---|---|
| `raw_grid_energy` | `3821438` | completed `0:0` | `/home/vault/c107fa/c107fa12/sequence-editing/runs/arc_jepa/arc_raw_grid_energy_firstpass_active_context` |
| `proposal_energy` | `3821439` | completed `0:0` | `/home/vault/c107fa/c107fa12/sequence-editing/runs/arc_jepa/arc_proposal_energy_firstpass_active_context` |
| `jepa_energy` | `3821440` | completed `0:0` | `/home/vault/c107fa/c107fa12/sequence-editing/runs/arc_jepa/arc_jepa_energy_firstpass_active_context` |

Setup: 120 ARC-AGI-1 train tasks, last 20 as task-level eval, 323 train
episodes, 64 eval episodes sampled from 20 eval tasks, 1500 steps, batch 16,
700 generated candidates per eval episode.

Results:

| Variant | Eval pass@1 | Oracle reachable | Pred distance | Oracle distance |
|---|---:|---:|---:|---:|
| `raw_grid_energy` | `0.0000` | `0.2083` | `95.19` | `15.94` |
| `proposal_energy` | `0.0000` | `0.2083` | `126.23` | `15.94` |
| `jepa_energy` | `0.0625` | `0.2083` | `129.35` | `15.94` |

Interpretation: the jobs started and completed successfully, but the learned
scorers are not useful yet. Candidate generation contains exact solutions for
about `20.8%` of eval episodes, while learned pass@1 is only `0-6.3%`.
Raw-grid energy gives the best mean predicted distance but selects no exact
targets; JEPA is the only nonzero exact pass@1 variant. The next ARC step
should fix candidate scoring supervision/eval before adding model complexity.

Rendered proposal/action examples are in
`../sequence-editing-report/assets/arc/diagrams/`, including PNGs and PDFs for
real ARC traces and one synthetic checkerboard copy-then-recolor trajectory.

## Structured JEPA Wave

Status: training/evaluation wave ended; results are partial and repair evals are prepared.

Purpose: test the next architectural hypothesis one component at a time after
single-CLS, predicted-goal, waypoint, Delta-JEPA, and verifier-free W/R waves
failed to produce non-oracle solves. This wave keeps the reliable full-grid
Sudoku base and adds structured latent slots, Delta-JEPA delta-source variants,
SD-JEPA-style progress subspace supervision, preference/action ranking, and a
goal+waypoint planner score. At the user's request it also includes combination
rows that stack LDAD, SD-progress, preference ranking, and waypoint/goal losses.

Scripts:

- `scripts/slurm/run_grid_goal_structured_wave_train.slurm`
- `scripts/slurm/run_grid_goal_structured_wave_eval.slurm`
- `scripts/experiments/submit_grid_goal_structured_wave.sh`

Submitted variants:

| Block | Variants | What it tests |
|---|---|---|
| Structured slots | `S0`-`S4` | 81 cells versus unit/global/progress/full slot layouts. |
| Delta-JEPA | `DJ0`-`DJ5` plus paired `_single` rows | Action conditioning crossed with all-token versus changed-cell+unit LDAD sources; every full-grid Delta row has a learned-CLS single-latent counterpart. |
| SD-JEPA | `SD0`-`SD3` | Separate progress projection and action-effect subspace. |
| Preference ranking | `PR0`-`PR4` | State progress rank, legal/listwise action rank, and predictor-successor ranking for PR2. |
| Goal/waypoint | `GW0`-`GW4` | Terminal goal, waypoint, waypoint+goal score, goal-conditioned waypoint, and multi-waypoint sketch. |
| Combinations | `C0`-`C7` plus required `_single` Delta pairs | LDAD+SD, LDAD+ranking, SD+ranking, LDAD+SD+ranking, and waypoint combinations. |

Submitted 46 training jobs and 46 dependency-held individual eval jobs. Initial
submission `3819274`-`3819365` failed at Hydra startup because newly added
model keys were missing from `configs/puzzle/grid_goal_sudoku.yaml`; those
eval placeholders were canceled. The corrected submission used:

- Non-unit/full-slot train jobs: `3819405`, `3819409`, `3819411`,
  `3819415`, `3819417`, `3819419`, `3819421`, `3819427`, `3819429`,
  `3819431`, `3819433`, `3819435`, `3819437`, `3819469`, `3819473`,
  `3819479`, `3819483`, `3819487`, `3819491`, `3819495`.
- Unit/full-slot replacement train jobs: `3819499`, `3819501`, `3819503`,
  `3819505`, `3819507`, `3819509`, `3819511`, `3819513`, `3819515`,
  `3819517`, `3819519`, `3819521`, `3819523`, `3819525`, `3819527`,
  `3819529`, `3819531`, `3819533`, `3819535`, `3819537`, `3819539`,
  `3819541`, `3819543`, `3819545`, `3819547`, `3819549`.
- Dependency-held evals: matching even IDs from `3819406` through `3819550`
  after canceled stale rows were replaced.

Operational fixes after submission:

- Declared the structured-wave model keys in Hydra config and added regression
  coverage that structured-wave `model.*` overrides are declared.
- Unit/full structured-slot rows OOMed at micro-batch 4 with 16 branches and
  depth 8. They now default to micro-batch 2 and grad accumulation 4, preserving
  effective batch size 8.

Eval runs diagnostics first, including LDAD action-delta probes,
delta-locality probes, and SD-progress ordering probes; goal/waypoint rows
include the combined `predicted_waypoint_goal_raw_euclidean_distance` score.

Final observed state on 2026-07-10:

- 32 final step-5000 checkpoints exist; later high-cost rows timed out before
  producing final checkpoints.
- 18 variants produced 144 planner rows. `S0_cell_baseline` and full-grid
  `DJ0`-`DJ3` solve `8/8` under oracle-goal raw-L2 latent rollout.
- Evaluated single-CLS Delta rows and single-CLS combination rows remain
  `0/8`, generally at remaining Hamming `54-56`.
- Fourteen completed checkpoints produced no planner rows because structured
  slot masks were not expanded beyond 81 cell tokens. That bug is fixed;
  repair submission is pending.
- Several old dependency-held eval jobs remain in `DependencyNeverSatisfied`;
  they were not canceled during this audit.

The full-grid solve result is not evidence that LDAD itself caused the solve:
`S0_cell_baseline` also solves `8/8`. In `DJ2`/`DJ3`, predicted displacement
decodes actions at `1.0` accuracy while encoded target displacement action
accuracy is approximately `0.0`, exposing the predictor-displacement shortcut.

## Wide Single-CLS Oracle Probe

Purpose: test whether one-vector board latents fail only because `d_model=256`
was too narrow. These runs use `d_model=1024`, learned CLS board state, causal
history predictor, oracle raw-L2 planning only, and no predicted goal/value head.

Result: all four train jobs completed. All four eval jobs hit the 8h walltime
after writing three shallow rows each. No variant solved any board.

| Variant | Train | Eval | Rows | Best result |
|---|---:|---:|---:|---|
| `W0_ema_vicreg_d1024` | `3817223` completed | `3817224` timeout/partial | 3 | `0/8`, h `54.125`, symbolic re-encode depth 4 |
| `W1_ema_ldad_set_d1024` | `3817225` completed | `3817226` timeout/partial | 3 | `0/8`, h `31.25`, symbolic re-encode depth 4 |
| `W2_ldad_vicreg_set_d1024` | `3817227` completed | `3817228` timeout/partial | 3 | `0/8`, h `34.125`, symbolic re-encode depth 4 |
| `W3_ldad_only_set_d1024` | `3817229` completed | `3817230` timeout/partial | 3 | `0/8`, h `32.875`, symbolic re-encode depth 4 |

Interpretation: widening the single latent helps symbolic re-encode hamming for
LDAD variants versus EMA+VICReg, but latent rollout remains near random
(`h ~= 54-55`) and oracle planning still does not solve. The one-vector path is
not competitive with the full-grid oracle recipe yet.

## Verifier-Free Energy Repair Wave

Purpose: repair the learned verifier-free W/R planner after the first sweep
showed oracle raw-L2 solved but learned compatibility/progress scoring did not.
This wave tested sampling hardness, W/R/rank loss scale, energy projections,
predicted-latent calibration, and local same-parent action discrimination.

Result: 16 variants produced planner/diagnostic results or partial results.
15 variants OOMed during training; their dependency-held eval placeholders were
canceled after Slurm marked them dependency-never-satisfied. No completed
learned verifier-free variant solved any board.

Best planner rows:

| Variant | Rows | Best result |
|---|---:|---|
| `B0_current_E4` | 18 | `0/8`, h `48.5`, remaining-count latent rollout depth 1 |
| `B1_current_F0` | 18 | `0/8`, h `51.75`, remaining-count symbolic re-encode depth 8 |
| `S1_balanced_partials` | 18 | `0/8`, h `49.0`, remaining-count latent rollout depth 8 |
| `S2_near_solution` | 18 | `0/8`, h `48.125`, remaining-count symbolic re-encode depth 8 |
| `S3_recovery_states` | 18 | `0/8`, h `49.25`, remaining-count latent rollout depth 1 |
| `S4_hard_action_sets` | 18 | `0/8`, h `50.625`, verifier-energy symbolic re-encode depth 8 |
| `W1_energy_x3` | 18 | `0/8`, h `48.25`, verifier-energy latent rollout depth 4 |
| `W2_energy_x10` | 13 | `0/8`, h `48.875`, verifier-energy symbolic re-encode depth 4 |
| `W3_energy_x30` | 18 | `0/8`, h `48.75`, remaining-count symbolic re-encode depth 4 |
| `W4_rank_x3` | 18 | `0/8`, h `51.0`, compatibility-energy latent rollout depth 8 |
| `W5_rank_x10` | 18 | `0/8`, h `51.5`, remaining-count symbolic re-encode depth 4 |
| `G1_energy_projection` | 18 | `0/8`, h `46.875`, remaining-count latent rollout depth 4 |
| `L5_exhaustive_small` | 18 | `0/8`, h `49.0`, compatibility-energy latent rollout depth 4 |
| `C1_sampling_weight` | 18 | `0/8`, h `50.125`, remaining-count symbolic re-encode depth 4 |
| `C2_sampling_rank` | 11 | `0/8`, h `50.375`, compatibility-energy latent rollout depth 8 |
| `C4_energy_proj_rank` | 18 | `0/8`, h `50.25`, remaining-count symbolic re-encode depth 1 |

Key diagnostics:

| Variant | Compat AUC | Remaining Spearman | Remaining MAE | Successor top1/top5 |
|---|---:|---:|---:|---:|
| `B0_current_E4` | `0.493` | `0.968` | `1.314` | `0.375/0.688` |
| `S1_balanced_partials` | `0.531` | `0.973` | `0.499` | `0.250/0.563` |
| `S2_near_solution` | `0.663` | `0.975` | `0.495` | `0.250/0.750` |
| `G1_energy_projection` | `0.524` | `0.973` | `1.765` | `0.563/0.875` |
| `L5_exhaustive_small` | `0.524` | `0.971` | `0.924` | `0.313/0.688` |
| `C2_sampling_rank` | `0.512` | `0.970` | `1.169` | `0.938/1.000` |

Interpretation: the remaining-edit scalar can be fitted on held-out encoded
states, but the score is not giving robust search behavior. Compatibility W is
mostly near chance; only near-solution sampling moves AUC materially above
chance. Several variants improve same-parent successor ranking, especially
`C2_sampling_rank`, but that still does not translate into beam solve rate.

Failed/OOM groups:
- Geometry fine-tune: `G2`, `G3`.
- Predicted-latent calibration: `P1`-`P4`.
- Local action rank except the small exhaustive row: `L1`-`L4`.
- Combined high-pressure rows: `C3`, `C5`-`C8`.

Output root:
`/home/atuin/c107fa/c107fa12/sequence-editing-repair-20260706/runs`.
