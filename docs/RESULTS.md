# Results

Last updated: 2026-07-12 22:11 CEST

## Corrected Sequence Result

All 315 seven-family sequence runs have dynamics and v5 reprobes. The old
completion-conditioned shape comparison used `.20` nominal chance despite
empirical majority baselines `.22-.41`; v5 adds balanced accuracy and explicit
majority baselines. No JEPA group passes balanced chance, trained gain, and
majority in every seed. Interleaved/global-random trajectories improve
aggregate count/relations at z16/z32, while coherent blocked trajectories
usually lose count. Some z32 JEPA rows gain weak complete-object position, but
reconstruction is much stronger and shape remains at baseline.

The exact fixed-N reflected matrix is complete with 90 runs/30 groups and v6
probes. z64/mixed `1..2` shape gain `+.081` falls to `+.001` at exact N=2
with inconsistent seed signs. No exact-load group passes an all-seed learned
shape, position, velocity, or relation gate; at exact N=8 every width loses
balanced shape from initialization. Raw linear velocity probes fail, while v6
color-centroid displacement reaches R2 `.768` at exact N=8, so velocity is
observable but not linearly exposed in the CLS.

## Full Bottleneck Result

All deterministic reflected rows and balanced controls are complete. Tight
z2-z8 states do not bind multi-object shape/position/velocity. z64/N2 is the
only multi-object row with all-seed trained shape improvement (`.446` final),
but position/velocity remain negative. z4/N6 reliably learns aggregate count
and relation statistics. At z64/N8, count rises to `.656` while effective rank
falls to `13.4/64`.

Balanced z32 reconstruction reaches foreground IoU `.138-.155` and positive
bound-position R2 `.389-.452` in every group/seed; z4 remains background-
dominated. The capacity can encode layout, but JEPA discards it. Full artifact:
`../sequence-editing-report/assets/moving_objects/deterministic_full_v2_summary.md`.

The revised variable-load seven-family trajectory matrix is complete: 315
trainers `3838208`-`3838522`, diagnostics `3838543`-`3838857`, and corrected
v5 probes `3840034`-`3840348`. The exact-load confirmation is active:
trainers `3841078`-`3841245`, dynamics `3841266`-`3841433`, v6 probes
`3841434`-`3841497` and `3841499`-`3841602`, and six-hour watchers
`3841603`-`3841622`.

All 168 exact-load trajectory rows, dynamics evaluations, and v6 probes are
complete. No configuration learns bound shape across all seeds. Position is
different: z16/N8 passes the learned complete-position gate in 5/7 families,
z32/N4 in 5/7, z32/N8 in 6/7, and z64/N2 in 5/7. Matched variable-load counts
are 0/7, 1/7, 4/7, and 0/7. Mean exact/mixed position R2 is
`.073/.027`, `.173/.031`, `.136/.114`, and `.348/-.003` respectively.
Noisy repair preserves z64/N2 position (`.231`, delta `+.156`) but no shape
signal. Tight z2/z4 position stays negative. This is color-indexed compact
geometry, not a factorized or permutation-invariant object representation.

## Rate-Constrained Bottleneck Gate

Hard quantization is implemented on the single CLS encoding, EMA targets, and
all rollout states. A naive 32-bit z8/q16 smoke collapsed to one code even with
strong VICReg. The corrected objective adds a soft-assignment usage term while
prediction and probes still consume only hard codes. GPU smoke `3841658`
completed `0:0` and reached 229/256 observable codes, `7.75` bits joint
entropy, and `29.93/32` summed coordinate entropy after 200 steps.

The dependency-held 108-row matrix pairs quantized levels `2/4/16` with
continuous level-0 controls under the same objective across z2/z4/z8, exact
N2/N4/N8, and three seeds. Trainers are `3841787`-`3841798` and
`3841803`-`3841898`; no row uses a grid latent.
Barrier `3841802` completed; 102 trainers are running and 6 are
priority-pending.

## Deterministic Moving-Object Binding Result

All 54 JEPA confirmations and 36 original reconstruction controls completed
with v4 bound probes. z32/N8 decodes aggregate count/color and weak bound shape
classification, but every reflect/wrap/rotate row has negative bound position
and velocity R2; rotating angular R2 is also negative. Raw pixels retain strong
position readout. z4 temporal predictors beat identity without semantic
velocity. The current representation is not a bound object state.

The original reconstruction decoder obtained `.933-.966` accuracy by predicting
background and had effectively zero foreground IoU. Foreground/background-
balanced reconstruction is the valid control. Earlier endpoint table:
`../sequence-editing-report/assets/moving_objects/deterministic_combined_v1_summary.md`.

## Moving-Object Bottleneck Smoke

Largest prepared cell `latent_dim=64`, `max_objects=8`, seed 1707, one train
step: job `3834574`, A40, completed `0:0` in 22s with 1482 MiB peak GPU memory.
This validates generator/model/probe execution only. The corrected sampler
keeps the chosen object count fixed across collision retries; an earlier local
stress test exposed and removed a severe bias toward low-count scenes.

## Moving-Object Bottleneck Result

All 90 trainers `3834593`-`3834682` and 90 identity diagnostics
`3834739`-`3834828` completed `0:0`. At `N=8`:

| z | Count learned/raw | Shape R2 | Color R2 | Velocity R2 | Relation R2 | fg IoU | Predictor wins |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | `.249/.234` | `-.072` | `.057` | `-.008` | `-.234` | `.000` | `1/3` |
| 4 | `.380/.241` | `.077` | `.209` | `.012` | `-.003` | `.000` | `1/3` |
| 8 | `.424/.242` | `.082` | `.230` | `.003` | `.102` | `.000` | `1/3` |
| 16 | `.548/.251` | `.163` | `.487` | `.002` | `.133` | `.000` | `0/3` |
| 32 | `.580/.230` | `.191` | `.703` | `-.025` | `.116` | `.001` | `0/3` |
| 64 | `.644/.232` | `.169` | `.794` | `-.050` | `.106` | `.005` | `0/3` |

The model learns non-pixel static summaries, with different capacity optima for
shape/relations versus count/color. It does not learn velocity or robustly
outperform an identity latent rollout. Full tables are in the report repo at
`assets/moving_objects/bottleneck_v1_summary.md`.

Temporal-delta objective smoke `3834839` (`z=32`, `N=8`, one step) completed
`0:0` on A40 in 12s. This validates the extra online-future encoder path only.

All temporal trainers `3834849`-`3834872` and diagnostics
`3834947`-`3834970` completed `0:0`. `z=4,N=8` is the sole gate winner:
velocity R2 `.021/.013/.008` and predictor-over-identity wins in all seeds.
Larger z rows do not recover velocity and usually lose to identity. Full tables
are in `assets/moving_objects/temporal_delta_v1_summary.md` in the report repo.

## Moving-Object Transfer Result

Wrapped/rotating z4/N8 trainers `3834975`-`3834986` and diagnostics
`3835493`-`3835504` completed `0:0`. Temporal improves wrapped predictor wins
from `2/3` to `3/3` but lowers count `.429 -> .317`, shape R2
`.096 -> .031`, and velocity `.017 -> .012`. Both rotating objectives beat
identity `3/3`, but angular R2 remains negative. This rejects z4/N8 temporal
as a general semantic recipe; the next run restores latent-size and load axes
for wrap and rotation.

## Moving-Object Capacity Transfer

All 228 wrap/rotation capacity trainers `3835525`-`3835752` and diagnostics
`3835930`-`3836157` completed `0:0`. Increasing load hurts count at fixed z;
increasing z recovers count/color/relations but loses motion. At N8, wrap count
rises `.273 -> .627` from z2 to z64 while velocity falls to `-.044`; no z>=8
base row beats identity. Only temporal z4/N8 passes velocity+identity in every
wrap seed, and no rotating row learns angular direction. Color declines from
step zero throughout, so absolute static readout is not uniformly learned.

Identical z4/N8 reruns disagree despite matching configs. Results are
provisional until deterministic exact reruns pass.
The exact 500-step v4 duplicate reprobes show no binding: shape is near chance,
bound velocity/position R2 are negative, and raw pixels decode position much
better. This is an early checkpoint result; the deterministic 5k gate is active.

## Object Dynamics Prestage Result

The 12-job `semantic_mix` base-objective prestage completed successfully on
A40 (`3831078`-`3831100`, interleaved with structured repair jobs). It crossed
`cls64_r1/cls64_r8`, LR `{1e-4,3e-4,1e-3}`, and `{500,1500}` steps at seed
`1707`. Checkpoints and `metrics.jsonl` are under
`/home/vault/c107fa/c107fa12/sequence-editing/runs/object_dynamics`.

Endpoint changes versus the fixed step-0 encoder are summarized below; latent
std ratio is endpoint across-sample std divided by initialization std.

| Model | LR | Steps | Std ratio | Object count delta | Current object delta | Object-map fg mIoU delta | Rollout-invalid AUROC delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cls64_r1` | `1e-4` | 500 | `.325` | `-.004` | `-.012` | `-.007` | `+.113` |
| `cls64_r1` | `1e-4` | 1500 | `.336` | `.000` | `-.023` | `-.009` | `+.145` |
| `cls64_r1` | `3e-4` | 500 | `.576` | `-.012` | `-.004` | `-.002` | `+.103` |
| `cls64_r1` | `3e-4` | 1500 | `1.266` | `+.012` | `-.055` | `+.000` | `-.063` |
| `cls64_r1` | `1e-3` | 500 | `.225` | `-.031` | `-.035` | `-.011` | `+.160` |
| `cls64_r1` | `1e-3` | 1500 | `.934` | `+.008` | `-.031` | `-.026` | `+.014` |
| `cls64_r8` | `1e-4` | 500 | `.545` | `+.020` | `-.043` | `-.003` | `+.009` |
| `cls64_r8` | `1e-4` | 1500 | `.542` | `+.020` | `-.043` | `+.011` | `-.074` |
| `cls64_r8` | `3e-4` | 500 | `.365` | `+.035` | `-.043` | `+.004` | `+.039` |
| `cls64_r8` | `3e-4` | 1500 | `.325` | `+.020` | `-.047` | `+.021` | `+.093` |
| `cls64_r8` | `1e-3` | 500 | `.174` | `-.020` | `-.035` | `+.026` | `+.090` |
| `cls64_r8` | `1e-3` | 1500 | `.102` | `-.016` | `-.016` | `+.048` | `+.073` |

Interpretation: the original short prestage does not select a default. All endpoints lose
current-object and latent-delta object probe accuracy relative to step 0,
despite improvements in some map/grid and invalid-state metrics. The
`cls64_r8/1e-3` rows reduce latent std to `.174/.102` of initialization while obtaining
the lowest rollout loss and strongest object-map gain, demonstrating why loss
alone is a misleading selection criterion. The random step-0 encoder also
already beats the raw-grid linear control on several labels.

The 5000-step base jobs `3831210`-`3831215` and stability jobs
`3831216`-`3831227` completed `0:0`. Stable-slot v3 jobs
`3831509`-`3831534` re-probed those 18 checkpoints and eight EMA/SIGReg
replications. The repair was material: v2 sorted slots by each partial visible
bbox, so persistent objects could exchange IDs while growing.

Three-seed results at LR `3e-4` select `cls64_r8 + EMA` as the current
compromise: object count `+.102 +/-.052`, balanced current object
`+.038 +/-.046`, action object `+.010 +/-.016`, object-map foreground mIoU
`+.0047 +/-.0031`, grid foreground mIoU `+.0028 +/-.0024`, and invalid-state
AUROC `+.117 +/-.007`, all trained minus matched initialization. `r8 +
SIGReg` has stronger count gain (`+.164 +/-.056`) but loses action-object
(`-.053 +/-.021`) and spatial information on every seed.

Implementation verification:

| Check | Result |
|---|---|
| JEPA fidelity contract tests | all objective/trajectory/probe/HWM/baseline/launcher contracts pass |
| Complete repository suite | passes, no xfails |
| Named-objective Hydra smoke | base/LDAD/VICReg/SIGReg/EMA/reconstruction/joint+staged HWM/full-grid pass on CPU |
| Prestage jobs | 12 completed `0:0` |
| Full-grid A40 smoke | job `3831536`, batch 64, `0:0`, about 3.1 GiB peak GPU allocation |
| Current v4 GPU gates | `3832316`-`3832318`, all `0:0`; H16/grid/reconstruction peak 8376/5372/2798 MiB |
| Completed calibrations | length `3832365`-`3832400`; seed-1707 HWM `3832401`-`3832414`, all `0:0` |
| HWM d4 confirmation | `3832932`-`3832943`, all `0:0` |
| Corrected probe refresh | length/HWM jobs `3832957`-`3832981`, all `0:0` |
| Trajectory gate | 45 train + 90 dual probes, `3833013`-`3833147`, all `0:0` |
| Historical phase wrapper | 486 dry-run commands; now retired |

The audit corrected material semantics before submission: LDAD uses encoded
adjacent endpoints with shared end-to-end gradients; SIGReg is the projected
Epps-Pulley Gaussian test; `80/15/5` is an effective sampled-window mix;
counterfactuals are structured off-path edits; and hidden state-level ownership
plus stable canonical slots prevents probe identity swaps. Probe
metrics now include corruption severity, foreground-balanced segmentation,
raw-grid controls, and geometry-based surprise.

The Delta-JEPA source defines LDAD on adjacent encoded endpoints and one
executed action. The previous long-horizon sequence-decoder requirement was
incorrect. Flat and H8 LDAD rows now have paired CLS/full-grid configs, but no
phase jobs have been submitted.

Probe v4 is implemented and ran over all 26 legacy checkpoints. It adds
parts/inside, nonlinear controls, rollout count, process labels, train-selected
attention metrics, foreground-aware neighbors, and executed-grid HWM planning
diagnostics. Fixed-batch qualitative exports on seed-1707 `cls64_r8`
checkpoints are descriptive only. After excluding one-cell examples, per-example
oracle-best-head current-object IoU over eight multi-cell cases is `.195` at
initialization, `.282` for EMA, and `.337` for SIGReg. Four-query current-object
neighbor match is `.25` for initial, trained, and foreground-aware pixel
neighbors for both objectives: this panel provides no nearest-neighbor evidence
of learned semantics. Selected-sample latent rollout MSE is `.11457` for EMA
and `.02150` for SIGReg. Aggregate v4 fixed-head probes, not these selected
panels, are the decision source.

All v4 jobs `3832338`-`3832363` completed `0:0`. Three-seed balanced results
split by factor: r1/SIGReg has the largest static count gain
(`+.085 +/-.026`), r1/EMA the largest current-object gain
(`+.056 +/-.015`), and EMA has the strongest >=4-cell fixed-head attention
gain (`+.292` r1, `+.298` r8). All four rows gain rollout-count transfer
(`+.413` to `+.496`) but lose hidden process-provenance decoding (`-.038` to `-.138`).
The original foreground-aware neighbor metric is canonical-slot agreement, not
semantic identity. Inside decoding and
small-MLP count also fail to improve. This does not validate broad object
abstraction or a single objective winner.

The three-seed EMA length sweep is also complete. From 5k to 50k, balanced
count gain rises from `+.004` to `+.106` for CLS64 and `+.043` to `+.091` for
CLS128; rollout-count transfer reaches `+.470/+.482`, and fixed-head >=4-cell
attention reaches `+.501/+.505`. Those gains do not recover the intended
process-provenance decoding falls to `-.130/-.152`. That target uses hidden
trajectory kind and can assign different labels to observationally equivalent
paint transitions. The reported NN deltas `-.048/-.021` compare canonical
scene-local slot IDs, not semantic identities. CLS128/50k gains inside
decoding (`+.085`) and small-MLP count
(`+.026`), but with substantial inside variance (`.091`). This is evidence for
increasing state-manifold/readout organization, not broad object-process
abstraction, so the 486-job phase remains held.

The one-seed HWM macro sweep completed `0:0`. Joint d4 is the best combined
row (macro retrieval `.258`, low-level retrieval and exact retrieved-action
execution `.203`, subgoal L1 `.101`, endpoint MSE `.00124`); staged d4 reaches
subgoal L1 `.099` but only `.133` retrieval success. All six joint/staged
d4/d8/d16 rows have zero exact CEM executions. Confirmation jobs
`3832932`-`3832943` now add seeds `2707/3707` for low, joint-d4, and staged-d4.

The corrected refreshes `3832957`-`3832981` and `3832984`-`3833008` completed
`0:0`. Balanced latent process accuracy is `.305-.412` versus `.181-.184` raw
and `.167` majority, but every trained-minus-initial delta is negative
(`-.036` to `-.139`), showing a random-feature rather than learned advantage.
Shape NN delta is positive only for CLS64/15k (`+.010`), every color NN delta is
negative (`-.010` to `-.170`), and completion NN MAE worsens by
`+.008` to `+.058`. The corrected controls support no learned semantic-neighbor
or hidden-process-provenance emergence claim.

The result analyzer had two scientific reporting bugs: dependent reprobes were
dropped when inline step-0 probes were disabled, and campaigns with identical
model/objective/length metadata could be pooled across run families. Both now
have regression tests; generated Markdown also displays family and max steps,
so length and HWM rows are auditable.

The evaluator now accepts an explicit probe trajectory and the analyzer keeps
one result per probe distribution rather than dropping or pooling them. The
bounded 5k trajectory gate crosses five regimes with CLS-EMA, reconstruction,
and full-grid EMA controls at three seeds. Each `3833013`-`3833147` unit is
train/common-`semantic_mix`/in-domain. All 135 jobs completed `0:0`.

The trajectory gate rejects the current temporal-abstraction claim. On common
`semantic_mix`, reconstruction beats EMA on balanced count for object-blocked,
frontier, and interleaved (`+.081/.067/.056` versus `+.037/-.001/.036`) and
improves object-map mIoU by about `+.025` while EMA stays near zero. EMA's
consistent advantage is rollout-count transfer (`+.413-.469` versus
`+.242-.264`). In-domain count is strongest for interleaved/global-random under
both EMA and reconstruction, not the more coherent object-blocked/frontier
orders. Semantic shape and shape/color/completion NN factors do not improve.
Full-grid EMA loses `.484-.624` common grid foreground mIoU on every regime.
The data contains useful static object structure, but there is no JEPA-specific
temporal object-emergence result.

Three-seed HWM d4 confirmation also fails planning. Staging reduces endpoint
MSE/model bias to `.000045/.0049` versus joint `.000825/.0550`; joint retrieved
exact execution is slightly higher (`.154` versus `.133`). Both schedules have
zero CEM exact success and about `.060` Hamming on every seed. Hierarchy should
not be scaled without a planner/objective redesign.

## ARC First-Pass Training Results

Three ARC candidate-scoring jobs were implemented and completed after adding
explicit active masks for padded context grids:

| Variant | Job | Eval pass@1 | Oracle reachable | Pred distance | Oracle distance |
|---|---:|---:|---:|---:|---:|
| `raw_grid_energy` | `3821438` | `0.0000` | `0.2083` | `95.19` | `15.94` |
| `proposal_energy` | `3821439` | `0.0000` | `0.2083` | `126.23` | `15.94` |
| `jepa_energy` | `3821440` | `0.0625` | `0.2083` | `129.35` | `15.94` |

All jobs completed with exit `0:0` on `rtxpro6k`. Output root:
`/home/vault/c107fa/c107fa12/sequence-editing/runs/arc_jepa`.

Interpretation: the first actual training jobs ran, but the result is negative.
The generated candidate sets contain exact solutions for `20.8%` of eval
episodes, while learned pass@1 is only `0-6.3%`. Raw-grid energy selects
closer candidates on average but no exact targets; only the JEPA variant gets
nonzero exact pass@1. Next work should improve candidate-set supervision/eval,
especially same-episode listwise ranking, before launching broader ARC JEPA
sweeps.

Audit caveat: historical `jepa_energy` training included direct target-positive
records with no generating action in its transition loss. That teleport target
contaminates the dynamics interpretation. The code now masks such records from
JEPA dynamics while retaining them for energy supervision; the three historical
job metrics above have not been rerun after this fix.

## Structured-Wave Audit Result

The ended structured wave produced 144 planner rows from 18 variants.
`S0_cell_baseline` and full-grid `DJ0`-`DJ3` each solve `8/8` with oracle-goal
raw-L2 latent rollout, while all evaluated single-CLS Delta/combination rows
remain `0/8`. This does not isolate an LDAD benefit because the non-LDAD cell
baseline already solves.

Diagnostics reveal a stronger historical fidelity problem: Grid-Goal training
LDAD used predictor-produced displacement when context was present. `DJ2` and
`DJ3` therefore reach `1.0` predicted-delta action accuracy while
encoded-target delta action accuracy is approximately `0.0`. Historical
Grid-Goal “SIGReg” rows likewise used covariance whitening. Future training is
repaired to use encoded endpoint LDAD and projected Epps-Pulley SIGReg; the old
checkpoints and results are not retroactively paper-faithful.

Fourteen final step-5000 checkpoints had no planner result because structured
slot latents carried more than 81 tokens while planner masks remained length
81. Mask expansion is fixed. Repair evals `3831076`, `3831077`, `3831079`,
`3831081`, `3831083`, `3831085`, `3831087`, `3831089`, `3831091`,
`3831093`, `3831095`, `3831097`, `3831099`, and `3831101` are running on
A40. Every repaired checkpoint now solves `8/8`, remaining Hamming `0.0`, on
its first depth-4 oracle latent-rollout row. This proves the mask repair reaches
planning and preserves oracle geometry; it does not establish a structured,
LDAD, SD, preference, or waypoint benefit. Remaining rows are running.

## ARC CPU Coverage Scaffold

Implemented a non-neural ARC-AGI-1 state/action coverage probe before any ARC
model training. The scaffold includes variable-size ARC grids with `30x30`
padding masks, leave-one-out train episodes, deterministic proposal extraction,
a typed action renderer, and a bounded oracle coverage analyzer.

Full ARC-AGI-1 training taxonomy from the official 400 training tasks:

| Metric | Value |
|---|---:|
| Training tasks | `400` |
| Leave-one-out train episodes | `1302` |
| All train pairs same-shape | `262` tasks |
| At least one shape-changing train pair | `138` tasks |

Bounded coverage on the first 50 sorted training tasks, two episodes per task,
depth `1`, beam width `4`:

| Setting | Solved | Mean distance |
|---|---:|---:|
| no cell fallback, no oracle output shape | `18/100` | `18.60 -> 12.59` |
| no cell fallback, oracle output shape | `20/100` | `17.04 -> 11.76` |
| bounded cell fallback, oracle output shape | `21/100` | `17.04 -> 11.67` |

After adding full-grid/hole proposals plus scale/color-map/mask-render actions,
the same no-cell depth-1 slice kept the exact solve counts but improved mean
best distance:

| Setting | Solved | Mean distance |
|---|---:|---:|
| improved, no oracle output shape | `18/100` | `18.60 -> 11.93` |
| improved, oracle output shape | `20/100` | `17.04 -> 11.18` |

Rendered proposal/action examples are available in
`../sequence-editing-report/assets/arc/diagrams/` as PNG and PDF files. They
cover complete-corners, crop, copy-patch, rotate, shape-changing copy, a
mask-render near miss, and a synthetic checkerboard copy-then-recolor
composition.

Interpretation: the interface is concrete, and the first training jobs now
prove the training/eval path runs end to end. Current depth-1 coverage remains
narrow, oracle output shape helps, and bounded pixel fallback adds little in
this shallow setting. The improved action layer helps distance but not exact
depth-1 solve count. After training, the bottleneck is learned candidate
scoring: exact targets exist in `20.8%` of held-out candidate sets, but the
models select exact targets only `0-6.3%` of the time.

## Horizon-Length Ablation Partial Results

K1-K4 evals are complete. K8/K16 evals initially failed after the Delta decoder
addition changed the current model state dict; repaired eval jobs are running
as `3808345`-`3808348`. The first repair row for each K8/K16 variant is now
available.

| Variant | Rows | Best oracle | Best predicted |
|---|---:|---|---|
| `K1_uniform` | 4 | `0/8`, h `49.5`, d4 | `0/8`, h `48.125`, d4 |
| `K1_smooth_count` | 4 | `0/8`, h `49.5`, d4 | `0/8`, h `48.125`, d4 |
| `K2_uniform` | 4 | `0/8`, h `44.375`, d16 | `0/8`, h `48.375`, d4 |
| `K2_smooth_count` | 4 | `0/8`, h `43.875`, d4 | `0/8`, h `48.75`, d4 |
| `K3_uniform` | 4 | `0/8`, h `42.75`, d4 | `0/8`, h `48.125`, d16 |
| `K3_smooth_count` | 4 | `0/8`, h `42.5`, d16 | `0/8`, h `48.25`, d16 |
| `K4_uniform` | 4 | `0/8`, h `35.125`, d4 | `0/8`, h `47.25`, d4 |
| `K4_smooth_count` | 4 | `0/8`, h `31.75`, d4 | `0/8`, h `46.125`, d16 |
| `K8_uniform` | 1 so far | `0/8`, h `3.75`, d4 | pending |
| `K8_smooth_count` | 1 so far | `8/8`, h `0.0`, d4 | pending |
| `K16_uniform` | 1 so far | `0/8`, h `28.875`, d4 | pending |
| `K16_smooth_count` | 1 so far | `0/8`, h `4.375`, d4 | pending |

Interpretation so far: clean one-long-rollout K4 improves oracle latent
planning substantially over K1-K3. The first K8 repair rows are much stronger:
`K8_smooth_count` solves `8/8` with oracle raw L2 at depth 4, while
`K8_uniform` reaches h `3.75`. Predicted-goal rows for K8/K16 are still
pending.

## Clean17 and Macro-HWM Waves

Both active 2026-07-02 waves are complete. Full tables are in
`../sequence-editing-report/RESULTS.md`.

| Wave | Rows | Missing | Best result |
|---|---:|---:|---|
| `clean17` | `76/76` | `0` | `G_ic_field_only`, oracle raw L2, `mpc_beam`, depth 16: `0/8`, h `11.12` |
| `macro_hwm` | `40/40` | `0` | `D16_H4_16`, baseline `mpc_beam`, depth 16: `0/8`, h `22.50` |

Clean17 best predicted-goal row: `G_ic_field_plus_mse`, predicted raw L2,
`mpc_beam`, depth 4: `0/8`, h `46.00`.

Interpretation: neither wave produced solves. Exact K=8 Clean17 did not
preserve the earlier minimal-aux oracle geometry. Distance-field goal variants
improved oracle remaining Hamming but not predicted-goal planning. Macro-HWM
high-level CEM/MPPI was worse than flat primitive MPC; codebook initialization
did not repair high-level planning.

## Current H1 Recipe / Old-Local Fast Wave

Minimal-aux 5k single-factor wave is complete. Train array `3803494` and eval
array `3803495` both completed all 29 tasks cleanly. The final eval matrix has
`456 / 456` rows and no malformed rows.

Best rows:

| Variant | Goal/score | Depth | Result |
| --- | --- | ---: | --- |
| `geom_oracle_progress` | oracle raw L2 | 4 | `8/8`, h `0.0` |
| `goal_distance_field_distill` | oracle raw L2 | 4 | `8/8`, h `0.0` |
| `rank_pairwise_oracle_action` | oracle raw L2 | 4 | `8/8`, h `0.0` |
| `reg_sigreg` | oracle raw L2 | 4 | `8/8`, h `0.0` |
| `base` | oracle raw L2 | 4 | `7/8`, h `0.125` |
| `hier_l4_l16` | oracle raw L2 | 4 | `7/8`, h `0.125` |
| `reg_vicreg` | predicted raw L2 | 16 | `0/8`, h `32.6` |

Interpretation: the 5k minimal-aux recipe already recovers strong oracle
global latent-rollout planning. Predicted-goal planning remains the bottleneck:
all predicted-goal rows are still `0/8`.

Dense-horizon caveat: `dense_k*` was based on the same minimal-aux script base,
but it changed loss weighting as well as horizon. The base uses
`dense_rollout_all_steps=false` with `[1,4,8,16]`; `dense_k*` uses
`dense_rollout_all_steps=true` with only `[K]` and only starts rollouts where
the full K-step future exists.

Follow-up audit: `minimal_aux` trained hierarchy `[4,16]`, but its good rows
used `mpc_beam`, not `hierarchical_beam`. `q(c,H0,Ht)` and no-stopgrad goal
target variants are not isolated goal-head ablations because they send goal
loss gradients into the state encoder. Exact dense K=8 weighting scripts are
implemented but not submitted.

The H1 recipe first wave is superseded. Health oversight `3800223` ran and
made no submissions. Post-eval oversight `3800130` was canceled before it ran,
so no Wave 2 has been submitted.

Eval jobs cannot be extended past the 24h partition max. The matrix runner now
resumes safely, so future repair jobs can append missing rows after current
evals finish or time out.

Depth-32 H1 triage jobs were added: `3801426`/`3801427` are running for
completed checkpoints `0-3,5,6`; `3801461`/`3801460` wait on remaining train
tasks `7-16`; `3801428`/`3801429` wait on retry train `3800228_4`. These jobs
test `mpc_beam` symbolic+latent and `hierarchical_beam` latent at depth 32
with global normalized, global raw L2, and changed-cell raw L2 scores.

Update at 16:50 CEST: retry train `3800228_4` completed and its depth-32
triage evals are running. Old-local eval stopped at `1628/1984` rows after 24h
timeouts. The strongest new H1 result is `minimal_aux`: `10/10` with
`mpc_beam + symbolic_reencode` under oracle global distance, and also `10/10`
with `hierarchical_beam + latent_rollout` under oracle global normalized/raw
L2 distance at depth 32. Predicted-goal planning remains `0/10`.

Old-local fast stopped with `1628 / 1984` eval rows. Dense variants are fully
evaluated and solve `0/10`. The first nonzero solve signal is
`rank_listwise_both_action`:

- symbolic re-encode, oracle changed-cell raw L2, depth 1: `6/10`, remaining
  Hamming `0.4`
- latent rollout, oracle changed-cell raw L2, depth 4: `2/10`, remaining
  Hamming `2.4`
- latent rollout, predicted-goal best: `0/10`, remaining Hamming about `48.5`

Interpretation: dense horizon alone is not enough. The useful signal is coming
from old-local action conditioning plus stronger action ranking, while the
predicted-goal planner remains unusable in this partial pass.

## Historical Local-Value Audit

The old Sudoku local-action result lives in the Grid3A/3B runs, especially
`sudoku_jepa_5m_local_direct_weighted_rollout_n2`. The archived action path was
`action_injection: local_value`: add the digit/value embedding to the selected
cell latent. It trained for `5000` optimizer steps at LR `1e-4`, with batches
of `768-1024` one-step transitions rather than full trajectories. The old
model had `8.55M` params; common current H1 hierarchy configs are about
`37.5M`. The 100% result was from oracle-goal symbolic re-encoding/reset
planning (`64/64` and `128/128`), not uninterrupted latent rollout; no-reset
latent planning on the same run solved only `4/64` to `7/128`.

## H1 Debug / H1 Extra Snapshot

H1 debug training/eval is complete; H1-extra eval is still running.

| Group | Jobs | State | Best current result |
| --- | --- | --- | --- |
| H1 delta | train `3795127`, eval `3795128` | complete | `0/10` exact solves |
| H1 no-delta | train `3795143`, eval `3795144` | complete | no-delta `K16_LR5e4`, `mpc_beam`, oracle changed-cell raw L2, depth 4: `0/10`, rem Hamming `6.6` |
| H1 hierarchical add-ons | eval `3795248`, `3795249` | complete | best hierarchical row: `0/10`, rem Hamming `28.8` |
| H1-extra | train `3795246_0-10`, replacement `3795327_11`; eval `3795247`, replacement `3795328_11` | train complete, eval running | 443 partial rows, best `rank_pairwise_both_action`: `0/10`, rem Hamming `14.9` |

Predicted-goal planning remains poor in the controlled H1 reruns: the best
predicted changed-cell row for the best no-delta checkpoint is still remaining
Hamming `33.8`. The controlled H1 reruns have not reproduced the earlier
`H1_hierarchy_dense_l4_l16` exact-solve signal.

## Weekend Next-Wave Result

The oversight chain ran `goal_conditioning` and submitted `dense_horizon`
twice. Later stages did not run because timed-out dense eval jobs left
malformed trailing JSONL, causing oversight jobs `3780036`-`3780040` to fail
with `JSONDecodeError`.

| Stage/run | Train | Eval | Valid rows | Best result |
| --- | --- | --- | ---: | --- |
| `goal_conditioning/G0_context` | `3780027_0` completed | `3780028_0` completed | 40 | oracle changed-cell raw L2, depth 32: `1/10`, rem Hamming `8.7` |
| `goal_conditioning/G1_initial_current` | `3780027_1` completed | `3780028_1` completed | 40 | oracle delta-top1 raw L2, depth 4: `0/10`, rem Hamming `42.9` |
| `goal_conditioning/G2_initial_current_oracle_progress` | `3780027_2` completed | `3780028_2` completed | 40 | oracle changed-cell raw L2, depth 4: `0/10`, rem Hamming `31.2` |
| `dense_horizon/DK2` | `3782967_0` and duplicate `3784073_0` completed | `3782968_0` and duplicate `3784074_0` timed out | 65 | oracle changed-cell raw L2, depth 32: `0/10`, rem Hamming `36.6` |
| `dense_horizon/DK4` | `3782967_1` and duplicate `3784073_1` completed | `3782968_1` and duplicate `3784074_1` timed out | 65 | oracle delta-top5 raw L2, depth 32: `0/10`, rem Hamming `45.1` |
| `dense_horizon/DK8` | `3782967_2` and duplicate `3784073_2` completed | `3782968_2` and duplicate `3784074_2` timed out | 65 | oracle changed-cell raw L2, depth 32: `0/10`, rem Hamming `38.8` |
| `dense_horizon/DK16` | `3782967_3` and duplicate `3784073_3` completed | `3782968_3` and duplicate `3784074_3` timed out | 65 valid + 1 malformed | predicted changed-cell raw L2, depth 16: `0/10`, rem Hamming `48.0` |
| `dense_horizon/DK32` | `3782967_4` and duplicate `3784073_4` completed | `3782968_4` and duplicate `3784074_4` timed out | 65 valid + 1 malformed | oracle changed-cell raw L2, depth 64: `0/10`, rem Hamming `48.1` |

Dense-horizon predicted-goal rows all solved `0/10` and stayed near
`47.6-48.9` remaining Hamming. The weekend result therefore does not support
the conditional-goal or dense-horizon changes as implemented. The prior
`H1_hierarchy_dense_l4_l16` follow-up remains the strongest signal: `6/10`
under oracle changed-cell local scoring, but still `0/10` under predicted
goals.

## Implementation Pass

No new experimental results were generated in the implementation pass. The
code now supports the staged next wave described in
`docs/EXPERIMENT_PLAN.md`, including conditional predicted goals,
hierarchical beam, hierarchy-dense rollout supervision, delta-top-k score
probes, ranking-loss switches, and an optional primitive/macro policy prior.

Safe cleanup removed disposable caches and previously archived failed-run
scratch directories only. Checkpoints were not deleted.

## Current Result

Follow-up wave:

- All follow-up train/eval jobs completed after resubmitting the two memory
  heavy variants at batch 4.
- Follow-up outputs contain 336 planner rows across all six variants and
  checkpoint-time outputs contain 240 planner rows for the current best
  action-suite run at `20k,30k,40k,50k,60k`.
- The only nonzero solve signal is
  `H1_hierarchy_dense_l4_l16` with `mpc_beam` and
  `oracle_goal_changed_cell_raw_euclidean_distance`:
  - depth 4: `0/10`, remaining Hamming `1.7`
  - depth 16: `6/10`, remaining Hamming `0.5`
  - depth 32: `4/10`, remaining Hamming `1.3`
  - depth 64: `5/10`, remaining Hamming `1.3`
- The same variant with oracle raw Euclidean but not changed-cell scoring got
  close but did not solve: best remaining Hamming `6.8`.
- The same variant with normalized oracle goal distance stayed worse:
  best remaining Hamming `22.9`.
- All predicted-goal rows solved `0/10`; the best predicted follow-up row was
  still around `36.0` remaining Hamming.
- Categorical CEM and hierarchical CEM solved `0/10` everywhere. They were
  faster than exhaustive beam, but the sampled search was much worse at the
  same score modes.

Diagnostics:

- `H1_hierarchy_dense_l4_l16` has strong oracle symbolic action top-1
  (`0.5938`) but weaker latent-rollout oracle top-1 (`0.3438`) and predicted
  top-1 (`0.3438` symbolic, `0.25` rollout).
- `F0_dense_k16` has the best latent-rollout oracle top-1 (`0.7188`) but did
  not solve; its best oracle changed-cell beam row had `9.9` remaining
  Hamming.
- `F1_dense_k32_detach8` emitted h32 rollout diagnostics as intended, but
  performed poorly in planning. Its h32 rollout MSE was `0.0141`, while
  oracle action top-1 was only `0.0625`.
- The wider `S0_scale_d384_dense` increased effective rank (`81.9`) but did
  not improve solve rate or predicted-goal planning.

Interpretation: hierarchy plus dense future prediction is the first latent
rollout configuration in this wave that can solve Sudoku under an oracle,
changed-cell metric. That is a real positive signal for the dynamics/latent
rollout path. It does not yet validate predicted-goal planning: the goal
predictor/goal metric gap remains large enough that all predicted-goal solve
rates are still zero.

Action-conditioning/stability suite:

- Training rerun `3768285` completed all 96 checkpoints.
- Corrected eval reruns completed:
  - main `planner_eval_latent`: 96/96 complete matrices, 1728 rows
  - depth-64 `planner_eval_latent_depth64`: 96/96 complete matrices, 576 rows
- Solve rate is `0.0` across all action-suite rows.

Best action-suite signal:

- One config is qualitatively better than the rest:
  `R4_no_goal_nce/A6_affected_marker_delta/S4_ema_vicreg/D0_uniform`.
  It reaches remaining Hamming `5.8` with normalized oracle-goal distance in
  both the main sweep and depth-64 sweep.
- The same config is much worse with predicted goals: remaining Hamming `36.6`
  normalized and `35.1` changed-cell raw. Predicted goal quality is still a
  major bottleneck.
- `A7_local_action_feature_delta/S4_ema_vicreg/D1_affected` is the next best
  changed-cell oracle row, with remaining Hamming `9.0` in the main sweep, but
  it also has zero exact solves.

## Follow-Up Audit

Dense future-state prediction, hierarchy, categorical CEM, and hierarchical CEM
were reviewed before full follow-up submission. The hierarchy path has the
intended shared latent space, stride-specific high-level predictors, high-level
latent CEM toward the goal, and primitive CEM toward the first subgoal.

The audit blockers were fixed before submission:

- Categorical and hierarchical CEM cap lookahead by remaining blank cells.
- CEM sampling stops safely after a sampled sequence fills the board.
- Rollout diagnostics emit configured long horizons, including h32.

Verification before submission: `source scripts/env.sh && pytest -q` ->
`70 passed`.

## Previous Result

All 13 Grid-Token Goal-JEPA training ablations completed successfully at
60,000 optimizer steps on RTX Pro 6000.

The first dependency-held planner eval array `3748790` started after training
and all tasks failed immediately with exit `1` during checkpoint loading. The
failure was not a planning failure: PyTorch 2.6+ defaulted
`torch.load(..., weights_only=True)` and rejected numpy scalar metadata in the
local training checkpoint payload. The eval loader now uses
`weights_only=False` for these trusted local checkpoints, and a regression test
covers this exact metadata case.

Planner eval rerun `3749458` is now running on `rtxpro6k`. All 13 array tasks
started, and all ablations have emitted diagnostics, so the checkpoint loader
fix is verified in Slurm. After about 6h10m, every ablation has completed
3/64 planner rows: symbolic-reencode/oracle-goal/beam-width-1 at depths `8`,
`16`, and `32`. Solve rate is `0.0` so far. The full planner matrix is likely
to hit the 24h wall, but completed rows are flushed to JSONL and will be
preserved.

Submitted a small follow-up probe for larger beams and raw oracle distance:
jobs `3750392`-`3750395` on `M0_full`, `R4_no_goal_nce`,
`R1_no_context_masks`, and `R6_no_action_rank`. The probe uses 8 boards,
symbolic re-encode only, beam widths `4,16`, depths `8,16,32,64`, and compares
the current normalized oracle distance with raw unprojected oracle Euclidean
distance.

Interim at 08:44 CEST on 2026-06-18: full matrix `3749458` is still running at
about 23h52m and is close to the 24h limit. Every ablation has completed 7/64
rows, reaching symbolic-reencode/oracle-goal/beam-width-4/depth-32. Solve rate
is still `0.0` across completed full-matrix rows.

Probe jobs `3750392`-`3750395` hit their 12h time limits. Each preserved 3/16
rows: normalized oracle distance at beam-width 4/depths `8`, `16`, and `32`,
all with solve rate `0.0`. They did not reach raw oracle Euclidean rows because
the normalized rows ran first and were slow.

Submitted more parallel per-metric probes, one job per checkpoint and score
mode, all pending initially on `rtxpro6k`: `3751931`-`3751938`. Settings are
8 examples, symbolic re-encode only, beam widths `4,16`, depths `8,16,32,64`,
and 24h time limit. Output roots are
`planner_probe_metric_norm_bw4_16_8ex/` and
`planner_probe_metric_raw_bw4_16_8ex/` under each selected checkpoint run.

Additional fast raw-only probes submitted to get quicker raw-distance signal:
`3751943` (`M0_full`), `3751944` (`R4_no_goal_nce`), and `3751945`
(`R7_no_terminal_corrupt`). Settings are 4 examples, symbolic re-encode only,
raw oracle Euclidean distance only, beam widths `4,16`, and depths `8,16`.
They were initially pending behind the running per-metric probes.

Interim at 14:15 CEST: full matrix `3749458` timed out after 24h with 7/64
rows per ablation preserved and no solves. Per-metric probes `3751931`-`3751938`
are still running after about 5h26m. Raw Euclidean rows are now available:
`R1_no_context_masks` raw reached solve rate `0.125` on 8 boards at beam-width
4/depths `8` and `16`; normalized R1 remained `0.0`. Raw Euclidean also greatly
reduced remaining Hamming for `R4_no_goal_nce` and `R6_no_action_rank`, though
solve rate is still `0.0` there so far.

Added six eval-only task-agnostic oracle score modes and submitted an 18-job
metric sweep at 14:28 CEST. The sweep covers `M0_full`,
`R1_no_context_masks`, and `R4_no_goal_nce`; metrics are raw squared Euclidean,
raw cosine, raw L2+cosine hybrid, raw L2 progress/delta, changed-cell raw L2,
and projected unnormalized Euclidean. Settings are 4 examples, symbolic
re-encode only, beam width `8`, depths `16,32`, and 12h time limit. Jobs
`3753366`-`3753383` all started immediately.

Interim at 16:57 CEST: per-metric jobs `3751931`-`3751938` are still running;
fast raw-only jobs `3751943`-`3751945` timed out after preserving 3/4 rows. In
the metric sweep, `R1_no_context_masks` is the first checkpoint with clear
nonzero solve signal: raw squared Euclidean reached `0.25` solve rate on 4
boards at beam width 8/depth 16, and raw L2 progress/delta reached `0.25` at
depths 16 and 32. `M0_full` remains `0.0`; `R4_no_goal_nce` remains `0.0` but
raw squared/progress rows have much lower remaining Hamming than the normalized
metric.

Final metric-sweep result: the strongest symbolic-reencode row is `R4_no_goal_nce` with
changed-cell raw Euclidean distance, beam width `8`, depths `16` and `32`,
which solved `3/4` boards (`solve_rate=0.75`) with mean remaining Hamming
`1.0`. `R1_no_context_masks` solved `1/4` (`0.25`) under several raw metrics,
including changed-cell raw, hybrid, cosine, raw squared, and progress/delta.
`M0_full` remained `0.0` across completed metric-sweep rows. The older
per-metric probes `3751931`-`3751938` timed out after 24h with 5/8 rows each;
best there remained `R1_no_context_masks` raw at `0.125`.

Planner implementation update: predicted-goal versions of the raw metric probe
scores are now implemented, raw L2 progress/delta no longer triggers the
zero-distance early-stop path, symbolic re-encode planning batches candidate
board encodes per beam layer, and latent-rollout planning batches predictor
expansions per beam layer. The strong changed-cell rows above are encoder
geometry/symbolic-transition diagnostics, not learned latent world-model solve
results. Verification: `source scripts/env.sh && pytest -q` -> `53 passed`.

Latent-rollout timing probe `3755858` completed on RTX Pro 6000. It used
`R4_no_goal_nce`, one board, one score
`oracle_goal_changed_cell_raw_euclidean_distance`, beam widths `4,16,32`, beam
depths `4,8,16,32`, and skipped diagnostics. Total wall time was `18m09s`.
Per-board row times ranged from `9.15s` at width 4/depth 4 to `275.75s` at
width 32/depth 32; all 12 width-depth rows summed to `994.98s`.

Submitted full latent-rollout sweep jobs `3755904`-`3756007`:
104 jobs = 13 ablations x 8 metric families. Each job bundles oracle and
predicted goal variants, uses latent rollout only, beam width `16`, depths
`4,16,32`, `10` boards, and skips diagnostics. Initial Slurm state: all 104
running, with 16 on `rtxpro6k` and 88 on `a40`. Output root:
`$PUZZLE_JEPA_WORK_ROOT/runs/grid_goal_sudoku_<ablation>/planner_latent_bw16_d4_16_32_10ex/<metric>/`.

Interim at 13:45 CEST: `M0_full` and `R1_no_context_masks` completed all 8
metrics successfully. 85 original jobs are still running on `a40`. Three
original jobs failed with transient Hugging Face cache/file-lock stale-handle
errors and were resubmitted as `3757178`-`3757180`, now running on `rtxpro6k`.
Partial scan: 101 output files, 272 planner rows, 16 complete outputs, no
solves yet (`max solve_rate=0.0`).

Final latent-rollout sweep result: retries `3757178`-`3757180` completed
successfully. All 104 output files and all 624 planner rows are complete. Total
solves: `0`; max solve rate: `0.0`. Best mean remaining Hamming was
`R7_no_terminal_corrupt` changed-cell raw Euclidean with oracle goal at depth
`4`: `47.9`.

Postmortem probes show the failure is action-discriminative prediction, not
just average drift. `R7_no_terminal_corrupt` has very low h32 rollout drift
(`~0.00064`) but still cannot plan. `R4_no_goal_nce` symbolic changed-cell
ranking picked target-consistent actions on all probed states, while predictor
top-action agreement with symbolic ranking was `0%`. Git history shows older
Grid3 configs used `action_injection: local_value`, directly adding the action
value embedding to the selected target-cell token; current Grid-Token Goal-JEPA
uses only a separate action token.

Implementation review status:

- Active Slurm jobs were cancelled before the refactor.
- New Grid-Token Goal-JEPA model/data/train/eval/planner path is implemented.
- Action-rank positives are now sampled explicitly as target-consistent
  solution fills, independent of random dynamics trajectories.
- `R1_no_context_masks` zeros context values as well as masks, and
  `encode_context` is value-blind when masks indicate no-context mode.
- Model `forward` derives row/column/token counts from inputs instead of
  hard-coding `9x9/81`.
- Remaining legacy CLS/value/causal modules and old grid train/eval/analysis
  paths were removed from the active tree.
- Progress ranking now receives `oracle_mask`; by default it applies to no
  rows, and training passes the true successful-trajectory mask.
- Action ranking now compares distances of encoded symbolic successor boards
  `f_theta(T(s,a),H_c)`, not predictor rollout latents.
- Diagnostics now include predictor rollout drift by horizon, latent-rollout
  top-positive action accuracy, predicted-goal vs oracle-goal alignment,
  distance-vs-Hamming Spearman correlation, action margins by fill depth, and
  terminal corruption margins by corruption size.
- HRM/TRM scaffolding remains intentionally as future baselines.
- Action-rank training now samples rank states from valid trajectory frames,
  not only the initial puzzle state.
- Added temporal straightening as a default geometry loss with ablation
  `R9_no_temporal_straightening`.
- Temporal straightening now matches the paper's curvature objective: it
  compares adjacent latent velocity vectors from fully valid three-frame
  triplets, uses the full active grid-token latent, and is independent of the
  predicted goal.
- Added linear warmup plus cosine decay: peak LR `1e-4`, warmup `1000`,
  final LR `1e-5`.
- The submitted suite used full-trajectory batch `8`, no gradient
  accumulation, and 60k optimizer steps.
- Current verification: `source scripts/env.sh && pytest -q` -> `32 passed`.
- Additional verification: `source scripts/env.sh && python -m compileall -q
  puzzle_jepa configs tests` passed.
- Running `pytest -q` without `source scripts/env.sh` fails at collection
  because the default Python cannot import `torch`.

Planner runtime risk remains: the largest beam matrix settings expand many
unbatched successor scores and may exceed the 24h eval limit.

## Batch Probe

Submitted four RTX Pro 6000 `M0_full` batch probes:

- batch 64: job `3748744`, failed CUDA OOM after `00:00:35`
- batch 128: job `3748745`, failed CUDA OOM after `00:00:35`
- batch 256: job `3748746`, failed CUDA OOM after `00:00:35`
- batch 512: job `3748747`, failed CUDA OOM after `00:00:35`

Each probe used one `rtxpro6k` GPU and printed `nvidia-smi` samples to its log.
Even batch 64 reached roughly full 96 GB VRAM, so none of the requested
microbatch sizes fit on RTX Pro 6000.

Submitted smaller full-trajectory probes on RTX Pro 6000:

- batch 4: job `3748774`, fit, then canceled
- batch 8: job `3748775`, fit, then canceled after the full suite submission
- batch 10: job `3748776`, fit initially but near the VRAM ceiling, then
  canceled
- batch 12: job `3748777`, failed CUDA OOM after `00:00:23`
- batch 16: job `3748778`, failed CUDA OOM after `00:00:23`

Current fit boundary appears to be between 10 and 12 full trajectories on one
RTX Pro 6000.
Batch 8 early throughput is roughly 100 optimizer steps/minute.

Full experiment suite training result:

- Training array `3748789`: 13 ablations, 60k optimizer steps, batch 8, no
  grad accumulation, all completed.
- Dependency-held planner eval array `3748790`: failed immediately on the
  checkpoint-loader issue described above; no planning results yet.

## Legacy Result

The previous faithful LeWM/CLS/value-head reset is now legacy. Its main result
was negative for Sudoku planning geometry: exact symbolic and true-Hamming
oracle scoring could solve, but oracle latent distance and learned scalar
goal-distance scoring did not produce solves. That result motivated the current
full-grid goal-prediction architecture.
