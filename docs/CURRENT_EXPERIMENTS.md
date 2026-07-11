# Current Experiments

Source of truth: `../sequence-editing-report/CURRENT_EXPERIMENTS.md`.

Last updated: 2026-07-11 17:06 CEST

## Moving-Object Bottleneck Grid

Status: all 90 training jobs `3834593`-`3834682` and dynamics diagnostics
`3834739`-`3834828` completed `0:0`. This is now the active object
abstraction experiment. It uses only one learned CLS latent and never a grid of
latent states.

The generator renders `1..N` persistent rigid objects with hidden shape,
color, position, and velocity. Two observed frames make velocity identifiable;
objects move autonomously, reflect at boundaries, and reverse in place on
collisions. The visual Transformer width stays fixed at 64 while the projected
state bottleneck crosses `z={2,4,8,16,32,64}` and
`max_objects={1,2,4,6,8}` at seeds `1707/2707/3707`.

Frozen probes report count, shape/color/velocity histograms, pair relations,
pixel foreground decoding, rollout semantics, latent variance/effective rank,
raw-frame controls, and matched step-zero deltas. GPU smoke `3834574`
completed `0:0` on A40 in 22s at 1482 MiB peak GPU memory. It validates
execution only. Output root:
`/home/vault/c107fa/c107fa12/sequence-editing/runs/moving_objects`.

Manifest: `manifests/bottleneck_v1_steps5000.tsv`. Six-hour watcher jobs are
`3834684`-`3834703`; watcher `3834684` completed `0:0`. Aggregation is
manifest-scoped so the separate smoke run cannot enter learning summaries.

At maximum load `N=8`, final learned/raw count accuracy rises from
`.249/.234` at `z=2` to `.644/.232` at `z=64`; shape R2 peaks at `.191` for
`z=32`, color R2 reaches `.794` at `z=64`, and relation R2 reaches `.133` at
`z=16`. Foreground pixel IoU stays at `.000-.005`. This is evidence for a
compact static object summary rather than a pixel map. Velocity R2 is never
positive reliably, and the predictor does not beat identity persistence in a
majority of `N=8` seeds at any z. The next gate must force temporal deltas into
the latent before transfer to other trajectory families.

The metadata-free temporal-delta variance gate is implemented for
`z={4,8,16,32}` x `N={4,8}` x three seeds (24 jobs). Largest-cell smoke
`3834839` completed `0:0` on A40 in 12s.

The 24 temporal-delta jobs were submitted as `3834849`-`3834872`.
Their manifest is `manifests/temporal_delta_v1_steps5000.tsv`; dedicated
six-hour watchers are `3834873`-`3834892`.

All 24 trainers and diagnostics `3834947`-`3834970` completed `0:0`. Only
`z=4,N=8` passes the temporal gate: velocity R2 is positive in all seeds
`.021/.013/.008`, and predictor MSE beats identity in all three. It retains
count `.371`, shape R2 `.068`, relation R2 `.068`, and foreground IoU `.000`.
This narrow high-load/small-state recipe is selected for trajectory transfer.

The first paired transfer is implemented: wrapped translation and rigid
rotation+translation, each with base and temporal z4/N8 rows at three seeds
(12 jobs). Rotating worlds add a clockwise/counterclockwise frozen probe;
wrapped relation labels use toroidal distance. Rotating temporal smoke
`3834974` completed `0:0` on A40 in 14s.

All 12 transfer trainers `3834975`-`3834986` and diagnostics
`3835493`-`3835504` completed `0:0`; watcher jobs are
`3834987`-`3835006`. Temporal pressure improves wrap predictor wins from
`2/3` to `3/3` but reduces count `.429 -> .317` and shape R2
`.096 -> .031`; velocity does not improve. Both rotating rows beat identity
`3/3`, but angular R2 is negative (`-.253/-.247`). The z4/N8 temporal effect
is therefore not a general semantic transfer. Next, recover the bottleneck and
object-load axes within wrap and rotation before construction/repair adapters.
The prepared follow-up contains 228 single-CLS rows; largest rotating base
smoke `3835521` (`z=64,N=8`) completed `0:0` on A40 in 21s.
All trainers `3835525`-`3835752` and diagnostics `3835930`-`3836157`
completed `0:0`; six-hour watchers are `3835753`-`3835772`. At N8, larger z
recovers count/color/relations but loses dynamics; color declines from step
zero throughout, and only wrap temporal z4 passes velocity+identity in all
seeds. Rotation direction is never decoded reliably.

Nominal duplicate z4/N8 runs differ materially despite identical configs, so
capacity conclusions are provisional. Deterministic cuBLAS/PyTorch/attention
is now the default; exact paired reruns gate the next submission.
Determinism pairs `3836199`-`3836202` completed `0:0`; both have exact model
tensors (`max diff 0.0`), final metrics, and metric histories. Six-hour
watchers are `3836203`-`3836206`. A 54-row deterministic confirmation across
reflect/wrap/rotate, z4/z32, N4/N8, and paired z4 objectives is prepared.
All trainers `3836223`-`3836276` started; six-hour watchers are
`3836277`-`3836296`.

Construction/completion/repair adapters are implemented for seven ordering
families with visible-only semantic labels, explicit scene count, rollout
count, completion probes, and zeroed stationary motion targets. All 19 focused
tests and the full suite pass; 40 eight-object stress samples per family and a
one-step CPU Hydra smoke pass. The 420-row launcher is dry-run only and stages
families 60 jobs at a time; submission waits for deterministic confirmation.
Largest noisy-repair temporal smoke `3836318` (z32/N8, one step) completed
`0:0` on A40 in 29s with finite v3 probes and checkpoint.
Schema v4 adds color-indexed bound shape/position/velocity/angular/completion
for latent, raw-frame, and rollout controls. Existing checkpoints can be
reprobed through a manifest-driven matched-initialization evaluator.
GPU v4 reprobe smoke `3836346` completed `0:0` in 14s; confirmation dynamics
diagnostics are dependency-held jobs `3836351`-`3836404`.
Full v4 reprobes are dependency-held jobs `3836574`-`3836627`.
Full duplicate v4 reprobes `3836450`-`3836453` are exact. At 500 steps, bound
shape is near chance and bound velocity/position R2 are negative; raw position
R2 is positive. The 5k result remains pending.
A reconstruction-only single-CLS control is implemented and CPU-smoked. Its
36 prepared rows match reflect/wrap/rotate, z4/z32, N4/N8, and three seeds;
the pixel decoder does not introduce a grid latent.
Smoke `3836463` completed `0:0`; trainers `3836464`-`3836499` are running,
diagnostics are `3836522`-`3836557`, and watchers are `3836502`-`3836521`.
Dependent v4 reprobes are `3836632`-`3836667`.

The historical 486-job object phase remains unsubmitted, and its two launchers
now exit as retired because they contain full-grid latent rows.

## Object Dynamics JEPA Scaffold

Status: all audited implementation gates pass. Probe-v4 and the three-seed
5k/15k/50k EMA length sweep completed, as did the one-seed HWM macro sweep.
Longer training improves count, rollout transfer, and attention but degrades a
hidden-provenance process probe; the prior nearest-neighbor score compared
scene-local canonical slots, not semantics. HWM d4 confirmation and the bounded
dual-probe trajectory gate are now complete. Neither hierarchy nor temporal
object ordering passes its intended gate, so the 486-job phase remains held.

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

Experiment grids:

| Grid | Jobs | State |
|---|---:|---|
| Original prestage LR/steps | 12 | completed `0:0` |
| Base 5000-step extension | 6 | completed `0:0`, jobs `3831210`-`3831215` |
| EMA/VICReg/SIGReg stability triage | 12 | completed `0:0`, jobs `3831216`-`3831227` |
| EMA/SIGReg winner replications | 8 train + 8 original probes | trains complete; four original r1 probes failed compatibility and were superseded |
| Stable-slot v3 re-probes | 26 | completed `0:0`, jobs `3831509`-`3831534` |
| Full-grid batch-64 smoke | 1 | completed `0:0`, job `3831536` |
| Probe-v4 batch-64 GPU gates | 3 | completed `0:0`, jobs `3832316`-`3832318` |
| Probe-v4 legacy re-probe | 26 | all completed `0:0`, jobs `3832338`-`3832363` |
| CLS64/128 EMA length calibration | 18 train + 18 dependent probes | all completed `0:0`, `3832365`-`3832400` |
| HWM macro/schedule calibration | 7 train + 7 dependent probes | all completed `0:0`, `3832401`-`3832414` |
| HWM macro-d4 confirmation | 6 train + 6 dependent probes | all completed `0:0`, `3832932`-`3832943` |
| Corrected semantic probe refresh | 18 length + 7 seed-1707 HWM checkpoints | all completed `0:0`, `3832957`-`3832981` |
| Dual-probe trajectory gate | 5 data regimes x 3 controls x 3 seeds, common + in-domain probes | all 135 completed `0:0`, `3833013`-`3833147` |
| Phase trajectory/model/objective sweep | 486 dry-run commands | held/not submitted |

Original replication probes `3831380/82/84/86` failed because an unfinished
grid-only `delta_pool` was temporarily required when loading older CLS
checkpoints. The pooler is now grid-only; v3 jobs `3831527`-`3831530`
supersede those failures. Original r8 probes `3831388/90/92/94` completed but
are also superseded by v3.

Calibration trainers defer full probes to one dependent v4 job. All length
pairs `3832365`-`3832400` and seed-1707 HWM pairs `3832401`-`3832414`
completed `0:0`. Macro-d4 confirmation uses low/joint/staged train-probe pairs
`3832932/33`, `3832934/35`, `3832936/37` at seed 2707 and `3832938/39`,
`3832940/41`, `3832942/43` at seed 3707. All completed `0:0`. Outputs are under
`/home/vault/c107fa/c107fa12/sequence-editing/runs/object_dynamics`.

Prestage job map (`semantic_mix`, `base`, seed `1707`):

| Model | LR | 500 steps | 1500 steps |
|---|---:|---:|---:|
| `cls64_r1` | `1e-4` | `3831078` | `3831080` |
| `cls64_r1` | `3e-4` | `3831082` | `3831084` |
| `cls64_r1` | `1e-3` | `3831086` | `3831088` |
| `cls64_r8` | `1e-4` | `3831090` | `3831092` |
| `cls64_r8` | `3e-4` | `3831094` | `3831096` |
| `cls64_r8` | `1e-3` | `3831098` | `3831100` |

All checkpoints and metrics are under
`/home/vault/c107fa/c107fa12/sequence-editing/runs/object_dynamics`.

Endpoint changes versus each run's fixed step-0 encoder (`500 / 1500`):

| Model | LR | Latent std ratio | Object-count acc delta | Object-map fg mIoU delta | Grid fg mIoU delta | Rollout-invalid AUROC delta |
|---|---:|---:|---:|---:|---:|---:|
| `cls64_r1` | `1e-4` | `.325 / .336` | `-.004 / .000` | `-.007 / -.009` | `+.015 / +.020` | `+.113 / +.145` |
| `cls64_r1` | `3e-4` | `.576 / 1.266` | `-.012 / +.012` | `-.002 / +.000` | `+.014 / +.016` | `+.103 / -.063` |
| `cls64_r1` | `1e-3` | `.225 / .934` | `-.031 / +.008` | `-.011 / -.026` | `+.021 / -.009` | `+.160 / +.014` |
| `cls64_r8` | `1e-4` | `.545 / .542` | `+.020 / +.020` | `-.003 / +.011` | `+.021 / +.029` | `+.009 / -.074` |
| `cls64_r8` | `3e-4` | `.365 / .325` | `+.035 / +.020` | `+.004 / +.021` | `+.021 / +.029` | `+.039 / +.093` |
| `cls64_r8` | `1e-3` | `.174 / .102` | `-.020 / -.016` | `+.026 / +.048` | `+.038 / +.035` | `+.090 / +.073` |

The original 500/1500-step rows did not pass the object-emergence gate. The
5000-step extension and stability triage are complete, and the winner rows
were replicated at seeds `1707/2707/3707`. Class-balanced v3
trained-minus-initial results at LR `3e-4` are:

| Model/objective | dObject count | dCurrent balanced | dAction object | dObject-map fg mIoU | dGrid fg mIoU | dInvalid AUROC |
|---|---:|---:|---:|---:|---:|---:|
| `cls64_r1/ema` | `+.009 +/-.018` | `-.054 +/-.017` | `-.034 +/-.038` | `+.0023 +/-.0013` | `+.0083 +/-.0020` | `+.111 +/-.018` |
| `cls64_r1/sigreg` | `+.111 +/-.050` | `+.013 +/-.035` | `+.030 +/-.013` | `+.0009 +/-.0021` | `-.0018 +/-.0021` | `+.060 +/-.009` |
| `cls64_r8/ema` | `+.102 +/-.052` | `+.038 +/-.046` | `+.010 +/-.016` | `+.0047 +/-.0031` | `+.0028 +/-.0024` | `+.117 +/-.007` |
| `cls64_r8/sigreg` | `+.164 +/-.056` | `+.021 +/-.032` | `-.053 +/-.021` | `-.0061 +/-.0016` | `-.0041 +/-.0006` | `+.102 +/-.097` |

`r8/EMA` is the only row with positive mean changes on all listed factors and
low surprise variance. `r8/SIGReg` learns the strongest count abstraction but
consistently loses action-object and spatial information. VICReg remains
unstable, including a severe `r8/3e-4` seed-1707 failure. The historical phase
launcher required explicit prestage selection and is now retired.

Probe-v4 three-seed trained-minus-initial results at LR `3e-4` revise that
interpretation:

| Model/objective | dCount balanced | dCurrent balanced | dRollout count balanced | dAttention IoU, >=4 cells | dProcess provenance | dNN canonical slot |
|---|---:|---:|---:|---:|---:|---:|
| `cls64_r1/ema` | `+.022 +/-.016` | `+.056 +/-.015` | `+.416 +/-.023` | `+.292 +/-.122` | `-.138 +/-.018` | `-.018 +/-.023` |
| `cls64_r1/sigreg` | `+.085 +/-.026` | `+.048 +/-.022` | `+.496 +/-.027` | `+.105 +/-.200` | `-.085 +/-.043` | `+.021 +/-.014` |
| `cls64_r8/ema` | `+.004 +/-.019` | `+.009 +/-.017` | `+.413 +/-.020` | `+.298 +/-.118` | `-.038 +/-.038` | `-.021 +/-.008` |
| `cls64_r8/sigreg` | `+.062 +/-.013` | `+.033 +/-.012` | `+.453 +/-.027` | `+.047 +/-.076` | `-.056 +/-.029` | `+.002 +/-.022` |

There is no v4 winner. SIGReg gives the clearest balanced static-count gains;
EMA gives much stronger train-selected fixed-head attention and r8/EMA retains
the best invalid-state surprise gain (`+.141 +/-.070`). Rollout-count transfer
improves strongly, but absolute r8/EMA balanced accuracy is only about
`.40-.44`, and this primarily shows that trained predicted latents enter the
decoder's state manifold. Every row loses one-step process-provenance decoding, inside
relation decoding declines and small-MLP count does not improve. The old NN
column is canonical spatial-slot agreement across unrelated scenes, not a
semantic object match, so it cannot support either a positive or negative
semantic claim. The completed width/length calibration does not confirm v3's
`r8/EMA` compromise as a general object abstraction winner.

Completed EMA length results (`mean +/-` population std, trained minus matched
initialization) are:

| Model/steps | dCount bal. | dRollout count bal. | dAttention >=4 | dProcess provenance | dInside bal. | dNN canonical slot |
|---|---:|---:|---:|---:|---:|---:|
| `cls64/5k` | `+.004 +/-.009` | `+.409 +/-.012` | `+.327 +/-.040` | `-.059 +/-.064` | `-.025 +/-.055` | `-.015 +/-.020` |
| `cls64/15k` | `+.058 +/-.034` | `+.437 +/-.036` | `+.449 +/-.098` | `-.079 +/-.014` | `-.050 +/-.027` | `-.037 +/-.023` |
| `cls64/50k` | `+.106 +/-.019` | `+.470 +/-.024` | `+.501 +/-.084` | `-.130 +/-.042` | `+.003 +/-.039` | `-.048 +/-.012` |
| `cls128/5k` | `+.043 +/-.028` | `+.469 +/-.038` | `+.496 +/-.075` | `-.065 +/-.013` | `-.045 +/-.035` | `-.033 +/-.023` |
| `cls128/15k` | `+.056 +/-.023` | `+.465 +/-.036` | `+.458 +/-.063` | `-.107 +/-.022` | `+.049 +/-.067` | `-.024 +/-.021` |
| `cls128/50k` | `+.091 +/-.025` | `+.482 +/-.031` | `+.505 +/-.062` | `-.152 +/-.048` | `+.085 +/-.091` | `-.021 +/-.025` |

This is a tradeoff, not a passed emergence gate. Fifty thousand steps gives
the strongest count/rollout/attention signal and the CLS128 rows recover some
inside/MLP-count information, but hidden process provenance gets less linearly
decodable with length. That label distinguishes `build`, `complete`, and
`repair_fill` using hidden trajectory kind even when the observed paint
transition is equivalent; it is not an identifiable primitive-action label.
The full 486-job phase is therefore not justified by these results.

Probe evaluation now adds a raw grid+action provenance baseline and true
current-object shape/color/completion nearest-neighbor metrics. Refresh jobs
`3832957`-`3832981` completed `0:0` over all length and seed-1707 HWM rows:

| Model/steps | Process balanced latent/raw/majority | dProcess balanced | dNN shape | dNN color | dNN completion MAE |
|---|---:|---:|---:|---:|---:|
| `cls64/5k` | `.408/.181/.167` | `-.036` | `-.025` | `-.144` | `+.025` |
| `cls64/15k` | `.367/.181/.167` | `-.076` | `+.010` | `-.010` | `+.032` |
| `cls64/50k` | `.305/.181/.167` | `-.139` | `-.039` | `-.030` | `+.058` |
| `cls128/5k` | `.412/.184/.167` | `-.040` | `-.039` | `-.170` | `+.008` |
| `cls128/15k` | `.346/.184/.167` | `-.106` | `-.021` | `-.023` | `+.025` |
| `cls128/50k` | `.326/.184/.167` | `-.126` | `-.029` | `-.080` | `+.046` |

Lower completion MAE is better. Latent balanced process accuracy exceeds raw
and majority controls, but every trained-minus-initial delta is negative and
worsens with length; this is a random-feature advantage, not learned process
emergence. Shape has one tiny positive row, color never improves, and
completion-neighbor error always worsens. There is no learned semantic-neighbor
emergence signal.

The trajectory gate isolates the next causal question at 5k steps: data
`{object_blocked,frontier_build,interleaved_build,global_random,random_off_manifold}`
x `{cls128_r8/EMA,cls128_r8/reconstruction,grid128_r8/EMA}` x three seeds.
Every checkpoint gets a common `semantic_mix` probe and a separate in-domain
probe. Dataset job blocks are object-blocked `3833013`-`3833039`, frontier
`3833040`-`3833066`, interleaved `3833067`-`3833093`, global-random
`3833094`-`3833120`, and random-off-manifold `3833121`-`3833147`. Within each
three-job unit the IDs are train/common/in-domain. Outputs are under
`/home/vault/c107fa/c107fa12/sequence-editing/runs/object_dynamics` with run
names `<data>_<model>_<objective>_trajectory_gate_steps5000_seed<seed>`.
All 135 jobs completed `0:0`. On the common probe, CLS trained-minus-initial
results are:

| Training data | dCount EMA/recon | dObject map EMA/recon | dAttention >=4 EMA/recon | dRollout count EMA/recon |
|---|---:|---:|---:|---:|
| object-blocked | `+.037/+.081` | `+.002/+.026` | `+.448/+.460` | `+.469/+.262` |
| frontier | `-.001/+.067` | `+.001/+.025` | `+.441/+.472` | `+.432/+.264` |
| interleaved | `+.036/+.056` | `+.001/+.025` | `+.393/+.459` | `+.413/+.250` |
| global-random | `+.080/+.075` | `+.003/+.027` | `+.463/+.443` | `+.433/+.242` |
| random-off-manifold | `-.080/-.097` | `-.014/+.049` | `+.135/+.160` | `+.001/-.005` |

Reconstruction, not JEPA, wins static count on three of four object datasets,
spatial object maps on every object dataset, and attention on three of four.
EMA's consistent advantage is rollout-count transfer. In-domain count gains
are largest for interleaved (`+.153/+.170` EMA/reconstruction) and
global-random (`+.166/+.135`), not object-blocked (`+.052/+.036`) or frontier
(`+.033/-.007`). Shape and shape/color/completion NN deltas remain near zero or
negative. Thus coherent temporal object ordering is not the cause of the
observed count/attention gains; static object-manifold exposure and
reconstruction explain at least as much. Random-off-manifold training hurts
common count, confirming that object-containing states matter. Full-grid EMA
is a failed control: common grid foreground mIoU drops by `.484-.624` and
object-map mIoU by `.057-.119` on every regime.

The three-seed d4 confirmation also fails the planning gate:

| Schedule | Endpoint MSE | Model bias L1 | Macro retrieval | Retrieved exact | Subgoal L1 | CEM exact / Hamming |
|---|---:|---:|---:|---:|---:|---:|
| joint | `.000825 +/-.000417` | `.0550 +/-.0479` | `.173 +/-.061` | `.154 +/-.036` | `.110 +/-.009` | `.000 / .0602` |
| staged/frozen | `.000045 +/-.000011` | `.0049 +/-.0010` | `.191 +/-.048` | `.133 +/-.013` | `.109 +/-.010` | `.000 / .0600` |

Staging strongly reduces latent model bias, while joint training gives slightly
higher retrieved exact execution. Neither schedule produces one exact CEM
execution on any seed, so no hierarchy row should be scaled.

Future phase models require both common `semantic_mix` and in-domain probes;
common-only evaluation confounds representation failure with distribution
shift. `random_off_manifold` remains the pure-random-edit training control.

Prepared scripts:

- `scripts/slurm/run_object_dynamics_train.slurm`
- `scripts/experiments/submit_object_dynamics_prestage.sh`
- `scripts/experiments/submit_object_dynamics_stability_prestage.sh`
- `scripts/experiments/submit_object_dynamics_stability_replication.sh`
- `scripts/experiments/submit_object_dynamics_balanced_reprobe.sh`
- `scripts/experiments/submit_object_dynamics_length_calibration.sh`
- `scripts/experiments/submit_object_dynamics_hwm_calibration.sh`
- `scripts/experiments/submit_object_dynamics_trajectory_gate.sh`
- `scripts/experiments/submit_object_dynamics_phase1.sh`

Verification:

- All objective, trajectory, probe, hierarchy, baseline, and launcher contracts
  pass, including the eight former strict research-gap specifications.
- The complete repository run passes with no xfails.
- Slurm verification `3830903` completed `0:0` on `a0123` in 20s; its log is
  `logs/jepa-obj-verify-3830903.out`. Preflight `3830803` failed `127:0`
  before collection because the repo-local interpreter was unavailable on the
  compute node; `logs/jepa-audit-verify-3830803.err` records the failure.
- One-step Hydra CPU runs pass for base, LDAD, VICReg, SIGReg, EMA, H16,
  full-grid, full-grid H8+LDAD, reconstruction, joint HWM, and staged/frozen
  HWM configurations. Batch-64 A40 smoke
  `3831536` completed `0:0` with about 3.1 GiB peak GPU allocation.
- Current batch-64 v4 A40 gates all completed `0:0`: H16 completion
  `3832316` in 15s at 8376 MiB, full-grid H8+LDAD `3832317` in 20s at
  5372 MiB, and reconstruction `3832318` in 20s at 2798 MiB. Their run
  directories are respectively
  `completion_h_cls128_h16_base_gpu_smoke_v4_h16_completion_20260710`,
  `semantic_mix_h_grid128_h8_ldad_gpu_smoke_v4_grid_h8_ldad_20260710`, and
  `semantic_mix_cls128_r8_reconstruction_gpu_smoke_v4_reconstruction_20260710`
  under the object-dynamics output root.
- LDAD now decodes encoded adjacent-state displacement with a shared
  end-to-end encoder; SIGReg now uses projected Epps-Pulley Gaussian testing.
- Effective semantic/counterfactual/wrong sampling is tested at `80/15/5`.
  Counterfactuals are local wrong-color/outgrowth/erase alternatives rather
  than shuffled gold actions. Probe v3 uses stable scene-canonical object
  slots; v2 incorrectly re-sorted partial visible bboxes and could swap IDs.
- Frozen evaluation now includes visible object geometry/color/shape/relations,
  missing/overgrowth/wrong-color severity, balanced foreground grid/object-map
  decoding, latent-delta actions, rollout transfer, hierarchy chunks, latent
  rank/nearest neighbors, geometry-based off-manifold surprise, and matched
  raw-grid baselines on a fixed held-out set plus a step-0 baseline.
- Probe v4 adds connected parts and an explicitly sampled `inside` relation,
  linear-versus-small-MLP controls, rollout object-count transfer, correction
  and one-step process labels, train-selected fixed-head/multi-cell/future-extent
  CLS attention, foreground-aware nearest neighbors, autoregressive high-level
  prediction, macro retrieval, CEM goal/subgoal reachability, exact symbolic
  plan execution, and dynamic aggregation of every numeric probe metric.
- HWM rows now use a Transformer action-chunk encoder projected to a
  low-dimensional macro space, two coarse teacher-forced transitions, CEM over
  latent macro-actions, state-valid categorical low-level CEM, and top-down
  latent subgoal matching. Joint and paper-style staged/frozen H8 rows are both
  represented.
- A reconstruction-only encoder control and fixed-batch qualitative exports
  for attention and latent-versus-pixel nearest neighbors are implemented.
- Staged HWM checkpoint reprobes now reconstruct the actual pretrained
  low-level initialization rather than a fresh random baseline. Probe calls
  restore the active CUDA RNG, so evaluation cadence cannot perturb later
  SIGReg directions. The macro encoder uses the paper's CLS bottleneck.
- `transform_identity` retains the original hidden shape class across pixel
  transforms. Length-aware completion trajectories preserve a visible seed in
  every object and support the two-step H16 training horizon of 32 edits.
- Grid-Goal SIGReg now uses the same projected Epps-Pulley test, and Grid-Goal
  LDAD now decodes encoded endpoint displacement. Historical checkpoints were
  trained before these repairs and retain the old objective semantics.
- Full-grid cell-token baselines and paired CLS/grid LDAD rows now exist for
  flat and H8 configs. Delta-JEPA defines adjacent-state action decoding; the
  prior long-horizon action-sequence requirement was erroneous.

Phase submission is now gated on GPU validation and selecting train length,
macro dimension, and joint-versus-staged HWM from the prepared calibrations.
The full phase is nine datasets x 18 rows x three seeds (`486` jobs), with
common `semantic_mix` probes and paired CLS/full-grid Delta rows.

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
mask expansion is fixed. Repair jobs are running on A40 with output suffix
`planner_eval_structured_mask_repair_20260710`:

| Variant | Job | Variant | Job |
|---|---:|---|---:|
| `S1_unit_slots` | `3831076` | `S2_global_slot` | `3831077` |
| `S3_progress_slot` | `3831079` | `S4_full_slots` | `3831081` |
| `DJ4_marker_cell_units` | `3831083` | `DJ5_cross_cell_units` | `3831085` |
| `SD0_projection_only` | `3831087` | `SD1_progress_rank` | `3831089` |
| `SD2_action_subspace` | `3831091` | `SD3_progress_action` | `3831093` |
| `PR0_state_pair` | `3831095` | `GW1_waypoint_only` | `3831097` |
| `C0_full_ldad_sd` | `3831099` | `C2_full_sd_pr` | `3831101` |

All 14 were still running on A40 at 19:21 CEST (elapsed about 2h52m). Every checkpoint has emitted its
first row: `8/8`, remaining Hamming `0.0`, for depth-4 oracle latent rollout
(oracle waypoint for `GW1`, oracle goal otherwise). This validates the mask
repair and oracle geometry only; remaining score/transition/depth rows are
still running.
Eighteen stale `DependencyNeverSatisfied` evals were canceled as superseded.

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

Status: original training/evaluation wave ended; results are partial and 14
structured-mask repair evals are running.

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
  repair jobs `3831076`-`3831101` are running on A40.
- Eighteen superseded `DependencyNeverSatisfied` evals were canceled.

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
