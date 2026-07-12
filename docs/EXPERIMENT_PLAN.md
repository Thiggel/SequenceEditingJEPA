# Experiment Plan

Source of truth: `../sequence-editing-report/BACKLOG.md` and
`../sequence-editing-report/CURRENT_EXPERIMENTS.md`.

## Active Moving-Object Bottleneck Plan

The immediate experiment replaces the broad low-level edit sweep. A fixed-width
Transformer encodes two consecutive rendered frames into exactly one projected
CLS state. Autonomous latent rollout predicts subsequent two-frame contexts;
no exact pixel action and no grid-token latent are available.

Cross `latent_dim={2,4,8,16,32,64}` with object load
`N={1,2,4,6,8}` and three seeds. Keep the completed `1..N` mixtures for a
non-degenerate count probe, but pair them with exact-load rows setting
`min_objects=max_objects=N` so compression pressure is measured directly.
Objects carry shape, color, reflected velocity, and pair-relation labels used
only by frozen probes. Gate on trained-minus-initial semantic gains, semantic
R2 versus pixel foreground decodability, rollout transfer, and effective rank.
After identifying a reproducible capacity transition, repeat the selected
bottlenecks on other trajectory families. Do not submit any full-grid latent
experiment.

Dimension is not a literal information-rate constraint. The active rate
control hard-quantizes every encoded, target, and rollout CLS state. It crosses
`z={2,4,8}`, levels `{2,4,16}`, exact `N={2,4,8}`, and three seeds, with
matched level-0 continuous rows under the same strong-VICReg objective. This
yields capacities from 2 to 32 bits. Require noncollapsed held-out joint and
coordinate entropy before interpreting semantic probes. Treat current relation
probes as aggregate summaries; pair-specific relations and same-color identity
need a permutation-aware extension before an object-level relation claim.

The first grid found static object summaries but no reliable velocity or
predictor advantage over identity. Before transfer, add a metadata-free
temporal-delta variance objective and gate it at `N={4,8}` and
`z={4,8,16,32}`. Require positive held-out velocity R2 and predictor wins in
all three seeds; static semantics alone do not pass the motion-world gate.

The temporal gate selected only `z=4,N=8`, but its paired wrapped/rotating
transfer failed semantically: predictor consistency transferred, while
velocity/angular factors did not and static factors declined. Restore the
latent-size and object-load axes within wrap and rotation to test whether each
trajectory has a different capacity optimum. Then transfer evidence-backed
cells to construction/completion/repair through a two-frame sequence adapter.
Retain base and temporal controls; never use a grid of latent states.

The two-frame adapter now covers all seven construction/completion/repair
families. Its primary labels are visible count and visible object factors;
hidden final-scene count is reported separately. The full 420-row matrix is a
prepared ceiling, not an automatic submission: use the active capacity result
to select latent sizes/object loads, then submit dependency-staged 60-job
families.

Capacity duplicates exposed GPU nondeterminism. Before selecting those rows,
run two identical deterministic jobs for wrap-temporal z4/N8 and a rotating
control; require exact model tensors and metrics. Then rerun selected 5k rows
under deterministic kernels before submitting any sequence-family stage.
The exactness gate, full deterministic reflected matrix, and balanced controls
are complete. Tight bottlenecks do not induce object binding. z4/N6 learns
aggregate count/relations; z64/N2 weakly learns shape; no multi-object JEPA row
learns bound position or velocity. Balanced z32 reconstruction proves a single
vector can carry color-specific position, so the missing spatial state is an
objective effect rather than an impossible capacity requirement.
Selection must use v5 color-indexed binding metrics in addition to bags:
bound shape, position, velocity/angular direction, completion, raw controls,
and one-step rollout transfer. For construction trajectories, report shape and
position both over all visible objects and conditioned on at least 50%/100%
completion. Shape claims must pass balanced accuracy and the empirical majority
baseline; ordinary accuracy versus nominal `.20` chance is invalid.
The active selected sequence matrix contains 315 rows and stages 45 jobs per
family. It tests tight high-load, z4 count/relation and temporal rows, z16
relations, z32 JEPA versus valid reconstruction at N4/N6/N8, and z64/N2 shape
across all seven construction/completion/repair trajectories. Do not submit the
older 420-row ceiling.

The exact reflected matrix rejects bottleneck-pressure abstraction: no
all-seed learned bound factor passes, and the mixed z64/`1..2` shape gain
vanishes at exact N=2. This leaves trajectory order as the controlled question.
The exact-N selected confirmation is submitted as 168 rows, 24 per family. It
keeps z2/z4/z16 high-load, z4 temporal, z32 N4/N8 with a reconstruction
control, and z64/N2. Interpret only all-seed trained-minus-initial gains against
the correct empirical baselines.

## Strategic Reframing

The current Sudoku evidence should not be read as JEPA reasoning. Oracle
full-grid solves can be explained by a per-cell symbolic latent plus a supplied
solution latent; predicted goals, learned verifier/value energies, waypoints,
and single-vector latents remain the meaningful failures.

For maze/ARC/language, low-level edit dynamics such as setting a cell or
inserting a token are too trivial to be the core world-model test. The next
proposal is to define tasks around abstract solution/output structure:
HRM-style maze as input grid to optimal path grid or candidate-output
refinement, ARC as output-grid/object refinement, and language as latent
block/future-solution prediction or high-level edits. New JEPA sweeps should
first specify direct HRM/TRM/seq2seq and raw-grid/value baselines.

Concrete architecture sketch:

- Encode `(task context, current candidate output)` into cell/object/segment
  latents, not only one global vector.
- Use high-level actions where possible: object transform, output-region fill,
  section rewrite, proof-step proposal, critique repair, or tool call. Primitive
  pixel/token edits are fallback decoder actions, not the main reasoning action.
- Train a JEPA predictor from `(latent state, action, context)` to the
  target-encoder latent of the next improved candidate or a future solution
  state.
- Keep a generator/decoder separate: it proposes concrete edits or renders a
  latent/candidate state into pixels/tokens. JEPA supplies representation,
  rollout prediction, value/energy regularization, and search scoring; it does
  not replace generation by itself.

## Object Dynamics JEPA Emergence Plan

This branch is not trying to solve ARC yet. It asks a narrower question:
whether a LeWM-like compressed latent world model trained on low-level edit
dynamics can recover hidden object/process abstractions.

Training observations are only:

- a grid state;
- a low-level action `paint/erase/recolor(row, col, color)`;
- future grid states for rollout supervision.

Hidden object metadata is excluded from training and used only for generation
and probes. The first trajectory regimes are:

| Config | Role |
|---|---|
| `object_blocked` | easiest temporal grouping condition |
| `frontier_build` | local coherent growth |
| `random_within_object` | object grouping without frontier cues |
| `interleaved_build` | persistent objects with interleaved edits |
| `global_random` | weak temporal object signal negative control |
| `noisy_repair` | structured repair/editability condition |
| `completion` | non-empty partial-object completion |
| `transform_identity` | transformation/recolor identity preservation |
| `random_off_manifold` | pure random-edit negative control |

Prestage is before T1/T2/etc.: it sweeps LR and train length on
`semantic_mix`. The first full sweep then crosses trajectory regimes with
single-CLS rollout horizons, stability objectives, and hierarchy:

| Block | Configs |
|---|---|
| Phase 1 | five CLS rows plus `grid128_r8` with `base` |
| Phase 2 | `cls128_r8` with `ldad/vicreg/sigreg/ema`, plus paired `grid128_r8/ldad` |
| Phase 3 | three joint HWM rows, paired CLS/grid H8 LDAD, and staged/frozen H8 |
| Control | `cls128_r8` reconstruction-only encoder baseline |

Frozen evaluation now covers visible object count/current/next object, color,
shape, connected-part count, bbox, centroid, area, completion, missing/overgrowth/wrong-color
severity, pair relations, spatially canonical visible object maps, foreground-
balanced grid decoding, actual latent-delta action fields, rollout transfer,
hierarchy chunks, nearest-neighbor semantics, latent rank, and off-manifold
rollout-error/manifold-distance AUROC. The same state/object probes run on
one-hot raw grids, and a fixed held-out set plus step-0 encoder baseline makes
checkpoint curves comparable. Hidden labels never train the JEPA.
Probe v4 additionally measures linear-versus-small-MLP gaps, rollout object
count, balanced `inside` relations, correction-process chunks, CLS attention
with train-selected heads plus multi-cell/future-extent targets,
foreground-aware nearest neighbors, high-level rollout error, macro retrieval,
continuous macro CEM, state-valid categorical primitive CEM, exact symbolic execution,
model bias, and subgoal reachability.
All phase rows use the same `semantic_mix` probe distribution so trajectory
comparisons do not also change the evaluation data. The phase matrix includes
`random_off_manifold` as a pure-random-edit training control.

The former strict implementation gates all pass. HWM now has a Transformer
macro encoder with a low-dimensional bottleneck, two coarse transitions,
continuous macro CEM, low-level categorical CEM, and top-down latent subgoal
matching. The matrix includes both joint representation learning and a
paper-style staged row initialized from and freezing the low-level model. The
remaining pre-phase gate is empirical: run probe-v4 compatibility, choose
train length from `{5k,15k,50k}` across three seeds and CLS widths, and choose
macro dimension `{4,8,16}` plus joint/staged schedule. Delta-JEPA itself
decodes adjacent displacement, not a long-horizon action sequence.

HWM is deliberately a local fully observed adaptation: its coarse predictor is
Markov and chunks have configured fixed lengths. The source paper uses a causal
interleaved high-level model and permits variable-length waypoint segments.
Results must be labeled accordingly unless those axes are implemented and
ablated. The calibration also compares exact executed-grid outcomes, not only
latent self-consistency.

The first 12-job base prestage completed on 2026-07-10 but did not select a default.
Every endpoint reduced current-object and latent-delta object probe accuracy
relative to its fixed step-0 encoder, while latent variance, map decoding, and
invalid-state AUROC gave conflicting rankings. The 5000-step extension and
EMA/VICReg/SIGReg triage completed, followed by three-seed EMA/SIGReg
replication. Stable-slot v3 probes select `cls64_r8 + EMA`, LR `3e-4`, as the
best compromise. The full guarded phase now contains nine datasets x 18 rows x
three seeds (`486` jobs); it remains held until the new calibration jobs finish.

## ARC Concrete Plan

Dataset:

- Start with ARC-AGI-1 public training tasks only for development. Official
  ARC-AGI-1 has 400 public training tasks and 400 public evaluation tasks.
- Use task-level train/validation splits inside the 400 training tasks for
  iteration. Hold public evaluation for final checks only.
- Each task episode is `(context examples, query input, target output)`.
  Construct leave-one-out episodes from all solved examples available in a
  training task: one pair is the query target, the remaining pairs are context.
- Pad grids to `30x30` with active masks. Stage 0 uses same-shape tasks or
  oracle output shape. Later stages add explicit canvas/shape actions.

State and action sampler:

- State is the current candidate output grid for the query, plus context.
- Initial states: blank/masked canvas, copied query input when shape-compatible,
  background-only grid, direct-baseline candidate, and hard corruptions of the
  target output.
- Object proposals are deterministic first: color connected components,
  non-background components, color groups, bounding boxes, rows/columns/lines,
  rectangles, holes/enclosed regions, and components from context inputs,
  context outputs, query input, and current candidate.
- Selection is part of an action, not necessarily a separate first step.
  Actions reference proposal IDs and parameters.
- Initial action DSL: set canvas, copy/paste object, translate, rotate,
  reflect, recolor, delete object, fill region, draw line/rectangle, crop/pad,
  apply color map, and fallback set-cell/set-region.
- For each sampled state, enumerate or sample a candidate action set containing
  target-improving oracle repairs, same-family hard negatives with wrong
  object/parameter/color, and random plausible DSL actions.
- Apply every sampled action symbolically to produce `next_state`. JEPA trains
  on both useful and bad transitions; value/ranking decides which successors
  are desirable.

Value/energy targets:

- Do not rely on an oracle goal latent at inference. Use it only as an upper
  bound/probe.
- Train a context-compatibility energy `E(context, query_input, candidate)`.
  Positives are target outputs; negatives are hard corruptions and wrong-action
  successors.
- Train action ranking/listwise loss on sampled candidate actions using target
  improvement during training: all actions that best reduce target distance or
  repair a target object are positives.
- Train optional progress/value targets as normalized edit distance or
  object-level distance to the target. These are training labels and probes,
  not an inference oracle.
- Goal-latent prediction from `(context, query_input)` is an ablation, not the
  default planner score.

Representations:

- Primary representation: grid tokens plus global context token and optional
  object slots. Grid tokens are not considered a trivial success because
  inference does not receive the target output; the hard question is context
  compatibility and transformation induction.
- Ablate `grid_only`, `grid_global`, `grid_object_global`,
  `object_global_only`, and `single_cls`.
- Treat `single_cls` as a compression stress test, not the expected main path.

Experiment gates:

1. DSL oracle coverage: before training, measure how often the action set can
   reach or substantially approach the target under oracle ranking.
2. Direct baselines: HRM/TRM/seq2seq and raw-grid value/policy baselines on the
   same episodes.
3. JEPA representation value: frozen JEPA latents plus small heads must beat
   raw-grid heads in compatibility/action ranking or sample efficiency.
4. JEPA dynamics value: latent rollout planning must approach symbolic
   re-encode planning; otherwise the transition predictor is not useful.
5. Non-oracle solve value: context-compatibility/value/search must improve
   exact output pass@1/pass@2 on held-out training tasks before public eval.

Initial experiment grid:

| Block | Variants | Question |
|---|---|---|
| State/action coverage | DSL oracle, corruptions, direct candidate repair | Is the proposed action space sufficient before learning? |
| Baselines | Direct HRM/TRM, raw-grid policy/value, raw-grid energy | Does JEPA beat non-JEPA baselines? |
| Representation | `grid_only`, `grid_global`, `grid_object_global`, `object_global_only`, `single_cls` | Are object/global slots useful and is single-state viable? |
| Target dynamics | EMA target, EMA+VICReg/SIGReg, online Delta-JEPA+LDAD | Which JEPA target scheme avoids collapse and helps action ranking? |
| Energy/value | compatibility, progress value, action ranking, goal latent, combined | Which planner score works without target output at inference? |
| Planning | symbolic re-encode beam, latent rollout beam, policy-prior beam | Does the learned world model actually reduce search cost? |

Required diagnostics:

- exact output pass@1/pass@2, cell accuracy, and shape accuracy;
- DSL oracle reachable rate and average oracle repair depth;
- energy AUC/calibration on target vs hard negatives;
- action top-1/top-5 target-improvement accuracy;
- frozen-latent probe performance versus raw-grid probes;
- latent rollout drift versus re-encoded successor latents;
- object proposal recall against changed target regions;
- qualitative panels with context, query, target, candidate trajectory, top
  actions, and failure reason.

Implementation status on 2026-07-07:

- CPU scaffold implemented in `puzzle_jepa.data.arc`,
  `puzzle_jepa.data.arc_proposals`, `puzzle_jepa.data.arc_actions`, and
  `puzzle_jepa.eval.arc_oracle_coverage`.
- The first coverage probe is deliberately non-neural and target-independent
  except for oracle scoring. It supports runs with/without oracle output shape
  and with/without bounded `set_cell` fallback.
- First official ARC-AGI-1 train slice result: first 50 sorted tasks, two
  episodes per task, depth `1`, beam width `4`, no cell fallback, no oracle
  output shape solves `18/100`; with oracle output shape solves `20/100`;
  adding bounded cell fallback solves `21/100`.
- First training jobs are now complete: `raw_grid_energy`, `proposal_energy`,
  and `jepa_energy`. They validate the end-to-end training path but not the
  research hypothesis; learned pass@1 is only `0-6.3%` against an oracle
  candidate-set reachability of `20.8%`. Next ARC work should improve
  candidate-set supervision/eval and add listwise ranking before scaling JEPA.

## Structured JEPA Wave

Implemented; the original wave ended and 14 structured-mask repair evals are
running as jobs `3831076`-`3831101` (non-contiguous odd/even IDs).

Research questions:

- Do explicit row/column/box unit slots, a global slot, and a separate progress
  slot preserve the factorization that single-CLS latents destroyed?
- Does Delta-JEPA LDAD work better when the decoded displacement is selected
  from all tokens, only cell tokens, the changed cell, or the changed cell plus
  Sudoku unit slots? Every Delta row is paired with a learned-CLS single-latent
  run.
- Does an SD-JEPA-style progress projection separate content dynamics from
  goal-distance/progress ranking?
- Do preference/ranking losses improve the local branch discrimination that
  verifier-free W/R heads failed to turn into solves? PR2 uses predictor
  successor latents rather than re-encoded symbolic successors.
- Can terminal goal prediction and receding waypoint prediction help when the
  planner scores waypoint distance strongly and terminal-goal distance weakly?

Prepared scripts:

- `scripts/slurm/run_grid_goal_structured_wave_train.slurm`
- `scripts/slurm/run_grid_goal_structured_wave_eval.slurm`
- `scripts/experiments/submit_grid_goal_structured_wave.sh`

The original structured wave produced 32 final checkpoints and 144 planner
rows from 18 variants. Fourteen structured-slot checkpoints had no rows due to
an 81-token planner mask bug. The mask is repaired; the 14 active evals write
to `planner_eval_structured_mask_repair_20260710`. Historical checkpoints still
use the pre-audit predictor-displacement LDAD/covariance-SIGReg semantics.
All 14 repaired checkpoints emitted an initial `8/8`, hamming `0.0`, depth-4
oracle latent-rollout row. Remaining conditions are still running.

## Verifier-Free Compatibility / Progress Energy Plan

Implemented, audit-fixed, submitted.

Research questions:

- Can a learned compatibility energy replace the oracle solution latent during
  inference?
- Can a separate remaining-edit head provide progress without collapsing the
  dynamics latent?
- Does counterfactual successor ranking train the exact local discrimination
  needed by verifier-free MPC?
- Does a policy prior help search after the learned state score is calibrated?

Fixed base recipe:

- full `9x9` grid-token latent, not single-CLS
- dropout off
- EMA target encoder plus VICReg
- editable non-clue cells
- counterfactual branches
- dense K8 smooth/count rollout supervision
- affected-context dynamics weighting
- no terminal goal predictor, no waypoint predictor, no oracle goal score

Planned model additions:

- tokenwise `W`: wrong-commitment compatibility energy
- tokenwise `R`: remaining-edit / Hamming-to-solution head
- optional action prior over editable cell-value actions
- verifier-free planner score `alpha * W + beta * R - eta * log pi`

Planned diagnostics:

- W AUC and wrong-count MAE on same-fill and near-solution corruptions
- R MAE and Spearman correlation against editable distance to solution
- successor pairwise/listwise action-ranking accuracy on latent rollouts
- predicted-latent W/R calibration versus encoded symbolic successors
- no-verifier MPC solve rate, remaining Hamming, first wrong commitment, and
  action-evaluation count

Prepared scripts:

- `scripts/slurm/run_grid_goal_verifier_energy_train.slurm`
- `scripts/slurm/run_grid_goal_verifier_energy_eval.slurm`
- `scripts/experiments/submit_grid_goal_verifier_energy.sh`

Audit blockers fixed on 2026-07-06:

- `verifier_energy` MPC no longer encodes the oracle goal latent during setup.
- `_sample_rank_actions(..., allow_overwrite=True)` skips full filled-wrong
  sequence states no longer; overwrite mode samples states mismatched from the
  solved board.
- The listwise verifier-targeted policy prior ignores no-blank wrong boards, so
  it now adds the positive repair cell even when the board has no blanks.
- Single-latent compatibility loss can use count targets as BCE labels and go
  negative no longer; BCE labels are clamped to binary and count supervision
  remains separate.

Regression tests were added in `tests/test_grid_goal_jepa.py` and now pass.

Prepared variants:

| Variant | Purpose |
|---|---|
| `E0_base_oracle_sanity` | Preserve oracle raw-L2 baseline with no verifier heads. |
| `E1_compat_state` | Train only W on encoded states plus corruption negatives. |
| `E2_remaining_state` | Train only R on encoded states plus corruption states. |
| `E3_wr_state` | Train W+R on encoded states. |
| `E4_wr_predicted` | Add W/R supervision on one-step predicted successor latents. |
| `E5_wr_pairwise_rank` | Add pairwise successor ranking on predicted latents. |
| `E6_wr_listwise_rank` | Replace pairwise with listwise successor ranking. |
| `E7_wr_listwise_policy` | Add verifier-targeted policy prior and planning bias. |
| `E8_wr_no_counterfactual` | Remove counterfactual dynamics branches from the full scorer. |
| `E9_wr_no_corruption` | Remove synthetic corruption negatives from the full scorer. |
| `F0_full_score` | W+R, predicted-latent calibration, corruptions, listwise ranking. |
| `F1_full_policy` | F0 plus verifier-targeted policy prior. |

## Counterfactual Editable Weekend Wave

Prepared, not submitted.

Research questions:

- Does counterfactual branching teach the world model action dependence?
- Does allowing non-clue cell overwrites remove the irreversible fill-only
  geometry failure?
- Does receding-horizon waypoint prediction work better than one-shot terminal
  goal prediction?
- Do asymmetric/value metric heads improve non-oracle planning?
- Can Delta-JEPA work once action/data coverage are fixed?

Stages:

| Stage | Purpose |
|---|---|
| `S` | Data/action smoke tests: old data, counterfactual fill, editable repair, AdaLN marker, old-local conditioning. |
| `E` | EMA+VICReg base, hierarchy, and waypoint variants. |
| `D` | Delta-JEPA paired full-grid and single-CLS variants. |
| `V` | Asymmetric/value geometry variants. |
| `I` | Integrated winners, including paired Delta-JEPA if the Delta gate passes. |

Operational invariant: any Delta-JEPA row must be paired as `_grid` and
`_single`. This applies to the dedicated Delta stage and any later autonomous
follow-up or integrated Delta stage.
