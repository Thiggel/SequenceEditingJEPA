# Runbook

Last updated: 2026-07-13 19:27 CEST

Long-form handoff source of truth: `../sequence-editing-report`.

## Single-CLS Hierarchy/Rollout Sweep

Active manifest:
`$PUZZLE_JEPA_WORK_ROOT/runs/controlled_objects/manifests/controlled_hierarchy_rollout_v5_steps20000.tsv`.
Jobs `3850619`-`3850672` write to
`$PUZZLE_JEPA_WORK_ROOT/runs/controlled_objects/controlled_hierarchy_rollout_v5_steps20000/`.
The 21 independent flat rollout/lambda jobs release 33 same-seed hierarchy
stages through `afterok` dependencies. Aggregate with
`scripts/analysis/analyze_controlled_objects.py` after all three seeds of a
group finish.

The only active launcher is
`scripts/experiments/submit_controlled_objects_hwm.sh`. It emits exactly 54
single-CLS, non-LDAD rows and dry-runs unless `SUBMIT=1`. Do not rerun while
the manifest jobs are active. The Delta and fidelity launchers are retired and
exit 2; the former long-gate launcher was removed.

Canceled off-scope jobs `3850564`-`3850569` produced no result. `3850564` ran
for `00:01:34`; the rest never started.

## Historical Controlled Gates

Completed v1 manifest:
`$PUZZLE_JEPA_WORK_ROOT/runs/controlled_objects/manifests/controlled_hwm_v1_steps20000.tsv`.
Jobs are `3849807`-`3849879` excluding `3849826`; outputs are under
`$PUZZLE_JEPA_WORK_ROOT/runs/controlled_objects/controlled_hwm_v1_steps20000/`.
All 72 jobs completed. Aggregate with
`scripts/analysis/analyze_controlled_objects.py`; archived output is under
`../sequence-editing-report/assets/controlled_objects/`.

V2 manifest `controlled_fidelity_v2_steps5000.tsv` maps completed jobs
`3850221`-`3850274`; aggregate artifact is
`../sequence-editing-report/assets/controlled_objects/controlled_fidelity_v2_summary.md`.
No row passes learned planning.

V3 manifest `controlled_delta_identifiable_v3_steps5000.tsv` maps completed
jobs `3850409`-`3850444`; artifact is
`../sequence-editing-report/assets/controlled_objects/controlled_delta_identifiable_v3_summary.md`.
An accidental duplicate submission ran `3850448`-`3850449` byte-for-byte
identically; held duplicates `3850450`-`3850483` were canceled before starting.

The trainer writes `config.json`, `metrics.jsonl`, final `metrics.json`, and
`checkpoint.pt`.

## Active Moving-Object Sweep

Dry run with `bash scripts/experiments/submit_moving_objects_bottleneck.sh`;
set `SUBMIT=1` for the 90 single-CLS rows. The launcher writes its manifest
under `$PUZZLE_JEPA_WORK_ROOT/runs/moving_objects/manifests`. Schedule the
six-hour monitor by setting `MOTION_MANIFEST` to that absolute manifest path
and running `scripts/experiments/submit_moving_objects_oversight.sh`.

Aggregate with `scripts/analysis/analyze_moving_objects.py`. Historical
`submit_object_dynamics_{phase1,trajectory_gate}.sh` launchers are retired and
exit nonzero because they contain prohibited full-grid latent rows.

The follow-up temporal-delta grid is dry-run by default at
`scripts/experiments/submit_moving_objects_temporal.sh`; set `SUBMIT=1` only
after the temporal objective smoke and full suite pass.

The selected z4/N8 transfer launcher is
`scripts/experiments/submit_moving_objects_transfer.sh`; it pairs base and
temporal objectives on wrapped and rotating trajectories and is dry-run by
default.

After that cell failed semantic transfer, the follow-up launcher
`scripts/experiments/submit_moving_objects_capacity_transfer.sh` restores the
full base bottleneck/load axes for wrap and rotation and the bounded temporal
subset. It is dry-run by default and prepares 228 single-CLS jobs.
Submitted manifest: `capacity_transfer_v1_steps5000.tsv`; trainers are
`3835525`-`3835752` and six-hour watchers are `3835753`-`3835772`.
Diagnostics are `3835930`-`3836157`; all train/eval rows completed `0:0`.
Capacity rankings are provisional because identical configs did not reproduce.
New training defaults to deterministic cuBLAS, attention, and PyTorch kernels.
Exactness jobs `3836199`-`3836202` pass; artifact:
`../sequence-editing-report/assets/moving_objects/determinism_v1.json`.
Confirmation manifest: `deterministic_confirmation_v1_steps5000.tsv`;
trainers `3836223`-`3836276`, diagnostics `3836351`-`3836404`, and v4
reprobes `3836574`-`3836627` all completed; watchers are
`3836277`-`3836296`.
The matched reconstruction-control launcher is
`scripts/experiments/submit_moving_objects_reconstruction_confirmation.sh`
and is dry-run by default (36 rows).
Submitted controls `3836464`-`3836499`, diagnostics `3836522`-`3836557`, and
v4 reprobes `3836632`-`3836667` completed but the decoder collapsed to
background; watchers are `3836502`-`3836521`.
Balanced replacement manifest: `reconstruction_balanced_v2_steps5000.tsv`,
trainers `3837715`-`3837779`. Deterministic reflected completion manifest:
`deterministic_reflected_matrix_v2_steps5000.tsv`, 78 new trainers
`3837714`-`3837827` plus 12 reused endpoints. Their dependent diagnostics and
v4 probes are interleaved in `3837829`-`3838120`; all completed. Artifact:
`../sequence-editing-report/assets/moving_objects/deterministic_full_v2_summary.md`.
The selected sequence launcher is
`scripts/experiments/submit_moving_objects_sequence_selected.sh`. Active
manifest `sequence_selected_v2_steps5000.tsv` has 315 trainers
`3838208`-`3838522`, 45 dependency-staged jobs per family. Diagnostics are
`3838543`-`3838857`, v4 probes `3838858`-`3839172`, and v5 correction probes
`3840034`-`3840348`; all completed. V5 includes half-complete and
complete-object shape/position, slot counts, balanced shape accuracy, and
empirical majority baselines. Artifact:
`../sequence-editing-report/assets/moving_objects/sequence_selected_v2_summary.md`.

The exact-load launcher is
`scripts/experiments/submit_moving_objects_fixed_bottleneck.sh`. Completed
manifest `fixed_load_reflected_v1_steps5000.tsv` has trainers
`3840351`-`3840440`, dynamics `3840442`-`3840531`, original probes
`3840532`-`3840621`, and watchers `3840622`-`3840641`. It sets
`min_objects=max_objects=N`.
Mixed-load reflected v6 reprobes `3840816`-`3840905` completed `0:0`; artifact
`../sequence-editing-report/assets/moving_objects/reflected_mixed_v6_summary.md`.
All 90 fixed-load v6 replacements completed; artifact:
`../sequence-editing-report/assets/moving_objects/fixed_load_reflected_v1_summary.md`.

The unsubmitted sequence-family launcher is
`scripts/experiments/submit_moving_objects_sequence_transfer.sh`. Its 420-row
dry run covers seven construction/completion/repair families and stages each
60-job family after the previous family completes. Do not submit the ceiling
matrix until capacity-transfer results select rows.
The smaller exact-N launcher is
`scripts/experiments/submit_moving_objects_sequence_fixed_selected.sh`.
Submitted manifest `sequence_fixed_selected_v1_steps5000.tsv` has trainers
`3841078`-`3841245`, dynamics `3841266`-`3841433`, v6 probes
`3841434`-`3841497` and `3841499`-`3841602`, and six-hour watchers
`3841603`-`3841622`. It contains 168 exact-load single-CLS rows staged 24 per
trajectory family. All 168 trainers/dynamics/probes completed `0:0`. Artifact:
`../sequence-editing-report/assets/moving_objects/sequence_fixed_selected_v1_summary.md`.
The rate-constrained launcher is
`scripts/experiments/submit_moving_objects_rate_bottleneck.sh`. Manifest
`rate_bottleneck_v1_steps5000.tsv` has trainers `3841787`-`3841798` and
`3841803`-`3841898`, barrier `3841802`, dynamics
`3841899`-`3842006`, probes `3842007`-`3842114`, and watchers
`3842115`-`3842134`. Barrier `3841802` completed. V1 fails the code-usage gate
and must not be interpreted by nominal bits. Corrected validation jobs are
`3844323`-`3844334`, all complete `0:0` with nontrivial held-out codes.
Replacement manifest `rate_bottleneck_v2_steps5000.tsv` has trainers
`3844346`-`3844453`, dynamics `3844454`-`3844561`, probes
`3844562`-`3844669`, all complete `0:0`. Artifact:
`../sequence-editing-report/assets/moving_objects/rate_bottleneck_v2_summary.md`.
Control manifest `rate_controls_v1_steps5000.tsv` uses trainers
`3844831`-`3844842`, all complete with dependent diagnostics/probes. Artifact:
`../sequence-editing-report/assets/moving_objects/rate_controls_v1_summary.md`.
Selected transfer manifest `rate_transfer_v1_steps5000.tsv` uses
dependency-staged trainers `3844843`-`3845004`. Wrapped, rotating,
object-blocked, frontier, random-within-object, interleaved, global-random,
and completion are complete; noisy-repair runs. Partial
artifact:
`rate_transfer_v1_partial_summary.md`. Transfer has
dependent dynamics/probes and 20 six-hour watchers.
Largest sequence GPU smoke is `3836318` (noisy-repair temporal z32/N8),
completed `0:0` in 29s.
Reprobe a manifest with `scripts/experiments/submit_moving_objects_probe_eval.sh`;
each job writes `probe_eval_v6.json` beside its checkpoint and can depend on
its corresponding trainer with `DEPEND_ON_TRAIN=1`.

## Historical Object-Edit Surface

The completed **Object Dynamics JEPA** surface tested whether a
compressed single-CLS latent world model trained only on low-level grid edit
dynamics can recover hidden object/process structure. Hidden objects are used
by the generator and probes only; training sees grids plus
`paint/erase/recolor(row, col, color)` actions.

- Package: `puzzle_jepa/object_dynamics/`
- Trainer: `puzzle_jepa/train/object_dynamics.py`
- Config root: `configs/object_dynamics/`
- Slurm template: `scripts/slurm/run_object_dynamics_train.slurm`
- Prestage dry-run wrapper:
  `scripts/experiments/submit_object_dynamics_prestage.sh`
- Stability-prestage dry-run wrapper:
  `scripts/experiments/submit_object_dynamics_stability_prestage.sh`
- Replication/re-probe wrappers:
  `scripts/experiments/submit_object_dynamics_stability_replication.sh` and
  `scripts/experiments/submit_object_dynamics_balanced_reprobe.sh`
- Retired phase sweep wrapper:
  `scripts/experiments/submit_object_dynamics_phase1.sh`
- Length and HWM calibration wrappers:
  `scripts/experiments/submit_object_dynamics_length_calibration.sh` and
  `scripts/experiments/submit_object_dynamics_hwm_calibration.sh`

The Slurm template can use `a40,rtxpro6k,a100`. The 12-job prestage completed
as jobs `3831078`, `3831080`, ..., `3831100`; outputs are under
`/home/vault/c107fa/c107fa12/sequence-editing/runs/object_dynamics`. It did not
select a default: current-object and delta-object probes declined in every row,
and latent-variance behavior conflicts with map/surprise metrics. Base
5000-step jobs `3831210`-`3831215`, stability jobs `3831216`-`3831227`, and
replication trainers `3831379/81/83/85/87/89/91/93` completed. Stable-slot v3
re-probes `3831509`-`3831534` also completed. All former strict fidelity
specifications now pass. The `486`-job phase remains held: the completed length
grid improves count/rollout/attention with training length, while the old
process target uses hidden trajectory provenance and the old nearest-neighbor
score compares canonical slots rather than semantic object factors.

Calibration trainers disable inline full probes and use one dependent v4 probe
at the final checkpoint. This avoids repeating MLP, attention, and CEM work.
The complete repository verification passes; the maximum H16 data
contract is 32 edits and is tested for every trajectory config.

Batch-64 v4 GPU gates `3832316`-`3832318` completed `0:0` on A40. They cover
H16 completion plus executed-grid CEM probes, full-grid H8+LDAD, and the
reconstruction control; peak allocation was 8376/5372/2798 MiB.

All 26 legacy probes, 36 length jobs `3832365`-`3832400`, and 14 seed-1707 HWM
jobs `3832401`-`3832414` completed `0:0`. Joint macro-d4 is the best one-seed
retrieval/subgoal compromise, but every HWM CEM row has zero exact executions.
Confirmation train/probe jobs `3832932`-`3832943` added seeds `2707/3707` for
low-level, joint-d4, and staged-d4; all completed `0:0`. Both schedules retain
zero CEM exact success, so hierarchy fails its gate. The 486-run phase is still
not submitted. Corrected semantic probe refresh jobs
`3832957`-`3832981` and balanced refresh `3832984`-`3833008` completed `0:0`:
balanced process accuracy beats raw controls but declines from initialization,
and shape/color/completion nearest-neighbor metrics do not improve reliably.

The bounded trajectory gate `3833013`-`3833147` completed all 135 jobs `0:0`.
Reconstruction explains static count/map/attention gains at least as well as
EMA, global-random/interleaved beat object-blocked/frontier on in-domain count,
and full-grid EMA loses `.484-.624` common grid mIoU. EMA uniquely improves
rollout-count transfer, but not semantic object factors. Do not submit the
486-job phase; the next experiment must change the data/objective/planner
contract rather than scale this matrix.

Prestage comes before T1/T2/etc. It calibrates LR and train length on the
`semantic_mix` dataset. T1 itself is the `object_blocked` trajectory regime.

Smoke command:

```bash
source scripts/env.sh
PUZZLE_JEPA_WORK_ROOT=/tmp/puzzle_jepa_object_dynamics_smoke \
python -m puzzle_jepa.train.object_dynamics \
  --config-name train \
  data=object_blocked model=cls64_r1 objective=base \
  output_dir=/tmp/puzzle_jepa_object_dynamics_smoke/run \
  training.max_steps=1 training.batch_size=2 \
  eval.probe_train_samples=8 eval.probe_eval_samples=6 eval.probe_steps=2
```

The calibration wrappers do not submit by default. The old phase wrapper is
retired and always exits nonzero; no phase selection can now submit it.

Regenerate the object result summary:

```bash
source scripts/env.sh
python -m puzzle_jepa.eval.object_dynamics_results \
  --root "$PUZZLE_JEPA_WORK_ROOT/runs/object_dynamics" \
  --output-dir /tmp/object_dynamics_summary
```

Targeted verification:

```bash
source scripts/env.sh
pytest -q tests/test_object_dynamics.py tests/test_object_dynamics_fidelity.py \
  tests/test_object_dynamics_remaining_fidelity.py \
  tests/test_grid_goal_remaining_fidelity.py tests/test_delta_jepa_fidelity.py -rx
```

Structured-slot planner mask repair jobs:

```bash
squeue -j 3831076,3831077,3831079,3831081,3831083,3831085,3831087,3831089,3831091,3831093,3831095,3831097,3831099,3831101
```

These 14 jobs were running on A40 at 19:21 CEST. All emitted one `8/8` depth-4 oracle
latent-rollout row, validating the mask repair. They skip existing diagnostics and write to
`planner_eval_structured_mask_repair_20260710` below each structured-wave run.
Do not cancel them. Aggregate planner rows only after jobs finish; empty files
while a job is running are not results.

## Legacy Sudoku Surface

The previous active path was **Grid-Token Goal-JEPA** for Sudoku. It remains in
the repo for reproducibility and tests.

- Config: `configs/puzzle/grid_goal_sudoku.yaml`
- Model: `puzzle_jepa/models/grid_goal_jepa.py`
- Data sampler: `puzzle_jepa/data/grid_goal_sudoku.py`
- Trainer: `puzzle_jepa/train/grid_goal_sudoku.py`
- Diagnostics: `puzzle_jepa/eval/grid_goal_diagnostics.py`
- Planner matrix: `puzzle_jepa/eval/grid_goal_planner_matrix.py`
- Planner: `puzzle_jepa/planning/grid_goal_planner.py`
- Training Slurm array: `scripts/slurm/run_grid_goal_sudoku_ablation.slurm`
- Dependency-ready planner eval array:
  `scripts/slurm/run_grid_goal_sudoku_planner_eval.slurm`
- Follow-up train array:
  `scripts/slurm/run_grid_goal_followup_train.slurm`
- Follow-up planner eval array:
  `scripts/slurm/run_grid_goal_followup_eval.slurm`
- Current-best checkpoint-time eval array:
  `scripts/slurm/run_grid_goal_best_checkpoint_eval.slurm`
- Next-wave staged train array:
  `scripts/slurm/run_grid_goal_next_train.slurm`
- Next-wave staged eval array:
  `scripts/slurm/run_grid_goal_next_eval.slurm`
- Next-wave submit wrapper:
  `scripts/experiments/submit_grid_goal_next_wave.sh`
- Next-wave oversight:
  `scripts/slurm/run_grid_goal_oversight.slurm`,
  `scripts/oversight/submit_grid_goal_oversight.sh`, and
  `scripts/oversight/grid_goal_next_wave.py`
- H1 debug/extra train and eval:
  `scripts/slurm/run_grid_goal_h1_debug_train.slurm`,
  `scripts/slurm/run_grid_goal_h1_debug_eval.slurm`,
  `scripts/slurm/run_grid_goal_h1_debug_hier_eval.slurm`,
  `scripts/slurm/run_grid_goal_h1_extra_train.slurm`, and
  `scripts/slurm/run_grid_goal_h1_extra_eval.slurm`
- Active clean17 train/eval:
  `scripts/slurm/run_grid_goal_clean17_train.slurm`,
  `scripts/slurm/run_grid_goal_clean17_eval.slurm`, and
  `scripts/experiments/submit_grid_goal_clean17.sh`
- Active Delta-JEPA train/eval:
  `scripts/slurm/run_grid_goal_delta_jepa_train.slurm` and
  `scripts/slurm/run_grid_goal_delta_jepa_eval.slurm`

Current active/prepared jobs are documented in `docs/CURRENT_EXPERIMENTS.md` and
`../sequence-editing-report/CURRENT_EXPERIMENTS.md`. All previous
LeWM/CLS/value-head jobs were cancelled or completed before this reset.

## Next-Wave Weekend State

The 12-hour oversight chain advanced from `goal_conditioning` to
`dense_horizon`, but then stopped on malformed partial eval JSONL produced by
timed-out eval jobs.

| Stage | Train jobs | Eval jobs | Outcome |
| --- | --- | --- | --- |
| `goal_conditioning` | `3780027`, tasks `0-2`, completed | `3780028`, tasks `0-2`, completed | 120 valid planner rows. Best row solved `1/10` with oracle changed-cell local scoring; predicted-goal rows solved `0/10`. |
| `dense_horizon` | `3782967`, tasks `0-4`, completed | `3782968`, tasks `0-4`, timed out | Submitted by oversight at 2026-06-25 23:55. Partial rows were written. |
| `dense_horizon` duplicate | `3784073`, tasks `0-4`, completed | `3784074`, tasks `0-4`, timed out | Submitted again by oversight at 2026-06-26 11:55; it wrote the same run dirs. Current dense-horizon files contain 65-66 rows each, with malformed trailing JSON in two files. |
| later stages | none | none | Not submitted. Oversight jobs `3780036`-`3780040` failed with `JSONDecodeError`; stale pending oversight jobs `3780041`-`3780042` were canceled on 2026-06-29 after this audit. |

Current next-wave run root:
`$PUZZLE_JEPA_WORK_ROOT/runs/grid_goal_next_wave`.

Best valid weekend rows:

| Run | Valid rows | Best oracle row | Best predicted row |
| --- | ---: | --- | --- |
| `G0_context` | 40 | changed-cell raw L2, `mpc_beam`, depth 32: `1/10`, rem Hamming `8.7` | delta-top3 raw L2, depth 32: `0/10`, rem Hamming `42.2` |
| `G1_initial_current` | 40 | delta-top1 raw L2, depth 4: `0/10`, rem Hamming `42.9` | delta-top3 raw L2, depth 4: `0/10`, rem Hamming `48.1` |
| `G2_initial_current_oracle_progress` | 40 | changed-cell raw L2, depth 4: `0/10`, rem Hamming `31.2` | delta-top5 raw L2, depth 64: `0/10`, rem Hamming `48.9` |
| `DK2` | 65 | changed-cell raw L2, depth 32: `0/10`, rem Hamming `36.6` | delta-top3 raw L2, depth 16: `0/10`, rem Hamming `48.7` |
| `DK4` | 65 | delta-top5 raw L2, depth 32: `0/10`, rem Hamming `45.1` | normalized predicted, hierarchical beam, depth 16: `0/10`, rem Hamming `48.8` |
| `DK8` | 65 | changed-cell raw L2, depth 32: `0/10`, rem Hamming `38.8` | delta-top5 raw L2, depth 32: `0/10`, rem Hamming `47.6` |
| `DK16` | 65 valid, 1 malformed | changed-cell raw L2, depth 64: `0/10`, rem Hamming `48.3` | changed-cell raw L2, depth 16: `0/10`, rem Hamming `48.0` |
| `DK32` | 65 valid, 1 malformed | changed-cell raw L2, depth 64: `0/10`, rem Hamming `48.1` | normalized predicted, depth 4: `0/10`, rem Hamming `48.9` |

Interpretation: the weekend next-wave did not improve the predicted-goal
planner. Conditional predicted goals and the dense-horizon stage regressed
relative to the previous `H1_hierarchy_dense_l4_l16` oracle-local result.
The only exact solve was a single oracle-local `G0_context` board, not a
predicted-goal success.

## H1 Debug Sweep

The controlled H1 debug sweeps are complete; H1-extra evals are still running.

Completed:

- H1 delta train/eval: `3795127`/`3795128`, six variants, batch `8`, 45k
  steps, hierarchy `[4,16]`, context-only goal predictor, `affected_marker`,
  delta predictor, EMA+VICReg, no goal NCE, fixed seed `5204`.
- H1 no-delta train/eval: `3795143`/`3795144`, same six variants and seed,
  but `model.predict_delta=false`.
- H1 hierarchical-beam add-on evals: `3795248` and `3795249`.
- H1-extra train: `3795246_0-10` plus replacement `3795327_11`. The original
  `3795246_11` OOMed at batch `8`; the comparable replacement used batch `4`
  with grad accumulation `2`, effective batch `8`.

Running:

- H1-extra evals `3795247_0-7,9,10` and replacement `3795328_11`.
- `3795247_8` completed. `3795247_11` was canceled after the superseded OOM.

Latest H1 results:

- H1 exact solve rate is `0/10` across completed rows.
- Best H1 `mpc_beam` row: no-delta `K16_LR5e4`, oracle changed-cell raw L2,
  depth `4`, remaining Hamming `6.6`. Depths `16/32/64` are `6.9-7.0`.
- Best predicted-goal changed-cell row on the same checkpoint is remaining
  Hamming `33.8`.
- Hierarchical beam is worse on these checkpoints: best remaining Hamming
  `28.8`.
- H1-extra has 443 partial rows so far, no exact solves. Best partial row:
  `rank_pairwise_both_action`, `mpc_beam`, oracle changed-cell raw L2,
  depth `4`, remaining Hamming `14.9`.

Storage cleanup at 2026-06-29 14:10 CEST:

- `$HPCVAULT/sequence-editing` is about `226GB`, down from about `949GB`.
- Deleted intermediate `checkpoint-[0-9]*.pt` files from completed non-active
  Grid Goal runs only. Kept final `checkpoint.pt`, configs, metrics,
  diagnostics, panels, and planner outputs.
- Active H1 debug/extra run roots were not pruned.
- `$WORK/.cache` was removed, freeing about `11.8GiB`. `$WORK/sequence-editing`
  remains about `19GB`.
- Common config: seed `5204`, batch `8`, 45k steps, LR `1e-4`,
  `affected_marker`, `predict_delta=false`, EMA+VICReg, no goal NCE,
  context-only goal predictor, temporal straightening on, dense base rollout
  horizons `[1,4,8,16]`.
- Variants: `rank_oracle_progress`, `rank_both_progress`,
  `rank_no_progress`, `rank_pairwise_oracle_action`,
  `rank_pairwise_both_action`, `rank_listwise_pred_action`,
  `rank_listwise_both_action`, `rank_no_action`, `hier_l4`,
  `hier_l4_l16_l32`, `hier_l4_l16_shared`, and
  `hier_l4_l16_hier_dense`.

Implemented and verified:

- conditional predicted goal `q(c,H0,Ht)`, recomputed at each MPC root state
- dense rollout supervision over intermediate base-predictor futures
- recursive dense supervision for high-level hierarchy predictors
- hierarchy levels with either separate predictors or one shared
  level-conditioned predictor
- hierarchical beam planner: high-level beam proposes latent subgoals, then
  primitive beam plans to the first subgoal
- delta-top-k local score probes for `k=1,3,5`
- progress-rank target switches: predicted, oracle, both, or none
- pairwise/listwise action-rank switches
- optional policy prior over primitive legal actions and fused macro-actions;
  planning can bias beam scores using `model.policy_prior_planning_weight`

Stage submit command used:

```bash
GRID_GOAL_STAGE=goal_conditioning TRAIN_CONCURRENCY=3 EVAL_CONCURRENCY=3 scripts/experiments/submit_grid_goal_next_wave.sh
```

Stages supported by the wrapper:

`goal_conditioning`, `dense_horizon`, `hierarchy_levels`,
`predictor_delta_topk`, `ranking_losses`, `hierarchical_planning`,
`policy_prior`.

Oversight scheduling command used:

```bash
GRID_GOAL_STAGE=goal_conditioning OVERSIGHT_COUNT=10 OVERSIGHT_INTERVAL_HOURS=12 OVERSIGHT_SUBMIT_NEXT=1 OVERSIGHT_CLEANUP=1 scripts/oversight/submit_grid_goal_oversight.sh
```

Oversight only reports by default. Set `OVERSIGHT_SUBMIT_NEXT=1` to let it
submit the next stage after the current stage is complete. Set
`OVERSIGHT_CLEANUP=1` for safe cache/failed-run cleanup. Do not set
`OVERSIGHT_DELETE_CHECKPOINTS=1` unless checkpoint deletion is explicitly
intended.

Verification from this pass:

- `source scripts/env.sh && pytest -q tests/test_grid_goal_jepa.py` -> pass
- `source scripts/env.sh && pytest -q tests/test_grid_goal_plan_regressions.py`
  -> pass
- `source scripts/env.sh && pytest -q` -> pass
- `source scripts/env.sh && python -m compileall -q puzzle_jepa scripts tests`
  -> pass
- `bash -n` over the new Slurm and oversight scripts -> pass

## Slurm Snapshot

Current sequence-editing status at 2026-06-25 11:56 CEST:

- Active next-wave jobs are listed above. Previous follow-up jobs are complete.
- Follow-up training/eval is complete:
  - original train array `3776065`: tasks `0,2,3,4` completed; tasks `1,5`
    OOMed at batch 8 and were superseded
  - replacement train array `3776086`: tasks `1,5` completed at batch 4
  - follow-up eval arrays `3776066`, `3776068`, `3776069`, `3776070`,
    `3776087`, and `3776088` completed
  - checkpoint-time eval array `3776072` completed
- Result files:
  - follow-up trained variants: 336 planner rows across 6 variants, beam/CEM
    planners, depths `4,16,32,64`, and six oracle/predicted score modes
  - checkpoint-time sweep: 240 planner rows across checkpoints
    `20k,30k,40k,50k,60k`, beam/CEM planners, depths `4,16,32,64`, and six
    score modes
- Best follow-up signal:
  `H1_hierarchy_dense_l4_l16` with `mpc_beam` and
  `oracle_goal_changed_cell_raw_euclidean_distance` solved `6/10` boards at
  beam depth `16` and had mean remaining Hamming `0.5`. Depths `32` and `64`
  solved `4/10` and `5/10`, respectively.
- All predicted-goal follow-up rows still solved `0/10`. The best predicted
  follow-up row had mean remaining Hamming `36.0`, so predicted-goal geometry
  remains the main blocker.
- Categorical CEM and hierarchical CEM did not solve any boards in this wave.
- Corrected action-conditioning/stability evals also finished:
  - main `planner_eval_latent`: 96/96 complete files, 1728 rows, zero solves
  - depth-64 `planner_eval_latent_depth64`: 96/96 complete files, 576 rows,
    zero solves
  - best action-suite row remains
    `R4_no_goal_nce/A6_affected_marker_delta/S4_ema_vicreg/D0_uniform` with
    `oracle_goal_distance`, mean remaining Hamming `5.8`
  - best predicted action-suite row remains much worse: mean remaining Hamming
    `35.1` with changed-cell predicted goal distance

Action-conditioning/stability suite state:

- Original training array: `3760074`, `grid_goal_act_train`, `0-95%32`,
  partitions `rtxpro6k,a100`, 24h limit.
- Original outcome: training had 29 completed tasks and 67 failed tasks.
- Failure reason: CUDA OOM on 40GB A100 nodes at batch 8. RTX Pro 6000 tasks
  completed.
- Stale eval `3760099` was canceled because it was `DependencyNeverSatisfied`.
- Rerun training array: `3768285`, failed indices only
  `24-26,28-38,42,44-95%16`, originally partition `rtxpro6k`.
- Training rerun `3768285` completed all 96 checkpoints. Tail RTX Pro tasks
  finished on 2026-06-23 around 17:32-18:40 CEST; A100 task `76` finished at
  15:14 CEST after `17:35:55`.
- Replacement eval array `3768300` was canceled and replaced by split evals so
  completed checkpoints evaluate immediately:
  - `3770937`: original completed tasks `0-23,27,39-41,43%32`
  - `3770953`: completed rerun tasks `24-26,28-38,42,44%32`
  - `3770954`-`3771004`: per-task evals for rerun tasks `45-95`, each
    dependency-held with explicit `afterok:<train-task-jobid>`
  - mapping file:
    `logs/grid_goal_act_eval_split_afterok_submit_20260622_2020.tsv`
- Monitor:
  ```bash
  squeue -j 3775750,3775751
  ```
- Eval status at 2026-06-24 10:36 CEST: prior evals are complete, but the
  2026-06-23 corrected rerun/depth64 submissions passed comma-separated
  variables through `sbatch --export=...`, so Slurm split them at commas.
  Current main outputs have 96 files but only 51 complete 18-row matrices, 16
  two-row matrices, and 29 one-row matrices; current depth64 outputs have 96
  one-row matrices.
- Current depth-32 result in observed complete rows: `0` solves across 306
  rows. Best row is still `remaining_hamming_mean=5.8` for
  `R4_no_goal_nce/A6_affected_marker_delta/S4_ema_vicreg/D0_uniform` with
  `oracle_goal_distance`. The same config reaches `9.2` remaining Hamming with
  changed-cell raw oracle L2, but predicted-goal variants remain much worse
  (`35.1-36.6` remaining Hamming for the best predicted rows).
- Corrected main eval rerun: `3775750`, RTX Pro-only, 8h limit, reruns
  indices `0,1,2,5-13,24-26,28,30-35,37,38,42-44,46-60,66-68%16`.
  This uses environment inheritance rather than comma values in `--export` and
  should restore 18-row `planner_eval_latent` matrices.
- Corrected depth64 eval: `3775751`, RTX Pro-only, 12h limit, indices
  `0-95%16`, beam width `16`, beam depth `64`, max steps `81`, same six score
  modes, output `planner_eval_latent_depth64`.
- Output root:
  `$PUZZLE_JEPA_WORK_ROOT/runs/grid_goal_action_suite/grid_goal_action_<base>_<action>_<stability>_<dynamics>/`.
- Scripts:
  `scripts/slurm/run_grid_goal_action_suite_train.slurm` and
  `scripts/slurm/run_grid_goal_action_suite_eval.slurm`.

Follow-up scripts are now submitted; script details:

- `scripts/slurm/run_grid_goal_followup_train.slurm`
  - array `0-5%6`, 24h, variants `F0_dense_k16`,
    `F1_dense_k32_detach8`, `H0_hierarchy_l4_l16`,
    `H1_hierarchy_dense_l4_l16`, `S0_scale_d384_dense`,
    `S1_deeper_d256_dense`
  - all use the current best recipe: `R4_no_goal_nce`,
    `affected_marker`, `predict_delta=true`, `EMA+VICReg`, uniform dynamics
  - default max steps `45000`; override with `TRAIN_MAX_STEPS`
- `scripts/slurm/run_grid_goal_followup_eval.slurm`
  - array `0-17%12`, 24h, one row per follow-up variant and planner group
  - planner groups: beam, categorical CEM, hierarchical CEM
  - hierarchical CEM skips non-hierarchical variants
  - default eval: latent rollout, 10 boards, beam width `16`, depths
    `4,16,32,64`, six oracle/predicted score modes
- `scripts/slurm/run_grid_goal_best_checkpoint_eval.slurm`
  - eval-only sweep for existing best action-suite checkpoint at
    `20k,30k,40k,50k,60k`
  - planner groups: beam and categorical CEM

Submitted follow-up jobs at 2026-06-24 11:54 CEST, all on `rtxpro6k`:

- Training array `3776065`, `grid_goal_fu_train`, array `0-5%6`, 24h.
- Dependency-held eval arrays:
  - `3776066`, array `0-2%3`, dependency `afterok:3776065_0`
  - `3776068`, array `6-8%3`, dependency `afterok:3776065_2`
  - `3776069`, array `9-11%3`, dependency `afterok:3776065_3`
  - `3776070`, array `12-14%3`, dependency `afterok:3776065_4`
- Existing-best checkpoint-time eval array `3776072`, `grid_goal_ckpt_eval`,
  array `0-9%10`, 24h, no dependency.
- Initial training tasks `3776065_1` (`F1_dense_k32_detach8`) and
  `3776065_5` (`S1_deeper_d256_dense`) OOMed quickly at batch 8 on RTX Pro.
  Their stale eval arrays `3776067` and `3776071` were canceled.
- Failed partial run directories were preserved as:
  - `grid_goal_followup_F1_dense_k32_detach8_failed_3776065_20260624_115750`
  - `grid_goal_followup_S1_deeper_d256_dense_failed_3776065_20260624_115750`
- Replacement training array `3776086`, array `1,5%2`, 24h, `BATCH_SIZE=4`,
  partition `rtxpro6k`.
- Replacement dependency-held eval arrays:
  - `3776087`, array `3-5%3`, dependency `afterok:3776086_1`
  - `3776088`, array `15-17%3`, dependency `afterok:3776086_5`
- Current state after resubmission: `3776065_0,2,3,4` and checkpoint eval
  tasks `3776072_0-6` are running; `3776072_7-9` are pending for resources;
  replacement train array `3776086_[1,5%2]` is pending on priority with
  estimated start `2026-06-24T22:06:08`; eval arrays `3776066`,
  `3776068`, `3776069`, `3776070`, `3776087`, and `3776088` are
  dependency-held.
- Output roots:
  - follow-up training/eval:
    `$PUZZLE_JEPA_WORK_ROOT/runs/grid_goal_followups/grid_goal_followup_<variant>/`
  - checkpoint-time eval:
    `$PUZZLE_JEPA_WORK_ROOT/runs/grid_goal_action_suite/grid_goal_action_R4_no_goal_nce_A6_affected_marker_delta_S4_ema_vicreg_D0_uniform/planner_eval_checkpoint_<step>_<planner>/`

Local verification for the follow-up implementation:

- `source scripts/env.sh && python -m compileall -q puzzle_jepa tests`
- Prior compact Python smoke over all six follow-up configs and
  beam/CEM/hierarchy planner paths completed in `8.68s` on CPU, but was too
  shallow.
- Audit regressions fixed at 2026-06-24 11:50 CEST:
  - `test_categorical_cem_mpc_handles_horizon_longer_than_remaining_blanks`
  - `test_hierarchical_cem_mpc_handles_subgoal_horizon_longer_than_remaining_blanks`
  - `test_rollout_diagnostics_include_configured_long_horizons`
- Fixes:
  - beam, categorical CEM, and hierarchical CEM now cap lookahead by remaining
    blanks so the simulated future never exceeds editable cells left
  - CEM sequence sampling stops safely if a sampled sequence fills the board
  - rollout diagnostics now include configured horizons such as h32
- Verification:
  - targeted audit tests: `3 passed`
  - `source scripts/env.sh && python -m compileall -q puzzle_jepa tests`
  - `source scripts/env.sh && pytest -q` -> `70 passed`

RTX Pro 6000 batch probes for `M0_full` were submitted and all failed quickly
with CUDA OOM:

- `3748744`: batch 64, `logs/grid_goal_bs64_3748744.out/.err`
- `3748745`: batch 128, `logs/grid_goal_bs128_3748745.out/.err`
- `3748746`: batch 256, `logs/grid_goal_bs256_3748746.out/.err`
- `3748747`: batch 512, `logs/grid_goal_bs512_3748747.out/.err`

Each probe requested one `rtxpro6k` GPU and 24h, and sampled GPU utilization
with `nvidia-smi`. None of the requested microbatch sizes fit; batch 64 already
used roughly the full 96 GB VRAM before failing.

Smaller full-trajectory probes:

- `3748774`: batch 4, canceled after confirming it fit
- `3748775`: batch 8, canceled after confirming it fit and submitting the full suite
- `3748776`: batch 10, canceled after confirming it fit but was near the
  VRAM ceiling
- `3748777`: batch 12, failed CUDA OOM after `00:00:23`
- `3748778`: batch 16, failed CUDA OOM after `00:00:23`

Wrong trajectories have the same frame count as oracle trajectories:
`#editable cells + 1`; they differ only in using random fill values.
In a 512-example train sample, trajectory lengths were min 47, median 57, mean
56.94, max 65 frames. Batch 8 logged roughly 100 optimizer steps/minute early
in training. The 60k-step suite completed in about 1.8 to 10.8 hours per
ablation, depending mainly on the multi-step horizon ablation.

Full suite state:

- Training array: `3748789`, `rtxpro6k`, array `0-12%13`, completed all 13
  ablations at 60k steps.
- Planner eval array: `3748790`, `rtxpro6k`, array `0-12%13`, failed
  immediately during checkpoint loading because PyTorch defaulted
  `torch.load` to `weights_only=True` and rejected numpy scalar metadata in
  the trusted local checkpoint payload.
- Fix: `puzzle_jepa/eval/grid_goal_planner_matrix.py::load_checkpoint` now
  passes `weights_only=False`; regression coverage was added for checkpoint
  payloads containing numpy scalar metadata.
- Training overrides: `TRAIN_MAX_STEPS=60000`, `BATCH_SIZE=8`,
  `GRADIENT_ACCUMULATION_STEPS=1`, `LEARNING_RATE=1e-4`
- Logs: `logs/grid_goal_train_3748789_<task>.out/.err` and
  `logs/grid_goal_plan_3748790_<task>.out/.err`
- Planner eval rerun: `3749458`, `rtxpro6k`, array `0-12%13`, running on
  nodes `a2041` and `a2843`; logs are
  `logs/grid_goal_plan_3749458_<task>.out/.err`.
- Interim rerun check at 15:02 CEST: all 13 tasks are still running after
  about 6h10m; all ablations emitted diagnostics; every ablation has completed
  3/64 planner rows: symbolic-reencode/oracle-goal/beam-width-1 at depths `8`,
  `16`, and `32`. Solve rate is `0.0` so far. Completed rows are flushed to
  JSONL and will be preserved if the full matrix hits the 24h wall.
- Small raw-distance probe submitted at 16:07 CEST:
  - checkpoints: `M0_full`, `R4_no_goal_nce`, `R1_no_context_masks`,
    `R6_no_action_rank`
  - jobs: `3750392`, `3750393`, `3750394`, `3750395`
  - settings: 8 examples, symbolic re-encode only, beam widths `4,16`, beam
    depths `8,16,32,64`, scores `oracle_goal_distance` and
    `oracle_goal_raw_euclidean_distance`
  - outputs:
    `$PUZZLE_JEPA_WORK_ROOT/runs/grid_goal_sudoku_<ablation>/planner_probe_bw4_16_raw_oracle_8ex/`

## Verify

```bash
source scripts/env.sh
pytest -q
python -m compileall -q puzzle_jepa configs
bash -n scripts/slurm/run_grid_goal_sudoku_ablation.slurm
bash -n scripts/slurm/run_grid_goal_sudoku_planner_eval.slurm
```

Current verification after fixing the planner checkpoint loader:

- `source scripts/env.sh && pytest -q tests/test_grid_goal_jepa.py`:
  `13 passed`
- `source scripts/env.sh && pytest -q tests/test_grid_goal_plan_regressions.py`:
  `19 passed`
- `source scripts/env.sh && pytest -q`: `32 passed`
- `source scripts/env.sh && python -m compileall -q puzzle_jepa configs tests`:
  passed
- Real checkpoint smoke load:
  `grid_goal_sudoku_M0_full/checkpoint.pt` loads on CPU and reports
  ablation `M0_full`, max steps `60000`.

Previous verification after fixing final action-rank state sampling:

- `source scripts/env.sh && pytest -q`: `26 passed`
- `python -m compileall -q puzzle_jepa configs`: passed
- Slurm launcher syntax checks: passed
- Import check confirms 13 ablations and beam widths/depths:
  widths `1,4,16,64`; depths `8,16,32,64`

Regression tests in `tests/test_grid_goal_plan_regressions.py` now pass. They
cover:

- action-rank positives are target-consistent solution actions
- `R1_no_context_masks` removes context value conditioning
- model `forward` accepts non-9x9 active grid-token tensors
- legacy CLS/value/causal paths are removed from the active source tree

Second-pass regression tests in `tests/test_grid_goal_plan_regressions.py` now
pass. They cover:

- progress ranking ignores random non-solution trajectories unless
  `oracle_mask` marks them successful
- action ranking uses encoded symbolic successor boards, not predictor latents
- `puzzle_jepa/models/recursive.py` and `puzzle_jepa/models/layers.py` remain
  intentionally as future HRM/TRM baselines
- rollout and goal-alignment diagnostics are present

Final-review regression test now passes:

- training samples action-rank boards from valid trajectory states, not only
  `batch.boards[:, 0]`.

Temporal-straightening regression tests now pass. They cover:

- a two-frame sequence must have zero curvature loss
- a masked sequence with no fully valid three-frame triplet must have zero
  curvature loss
- changing only the goal must not change the curvature loss of a fixed
  encoded trajectory
- full active grid-token latents are used rather than only mean summaries

Operational risk:

- the largest planner matrix settings (`beam_width=64`, `beam_depth=64`) expand
  many unbatched successor scores and are likely to be very slow without
  batching or a branch policy.

## Eval Rerun

The training checkpoints are complete. Planner eval rerun `3749458` is active:

```bash
squeue -j 3749458
```

Each training task writes to:

```text
$PUZZLE_JEPA_WORK_ROOT/runs/grid_goal_sudoku_<ablation>
```

Each dependency eval task writes:

```text
$PUZZLE_JEPA_WORK_ROOT/runs/grid_goal_sudoku_<ablation>/planner_eval/
```

## Active Ablations

`M0_full`, `R1_no_context_masks`, `R2_mean_pooled_distance`, `R3_k1_only`,
`R3_k4`, `R3_k8`, `R3_k16`, `R4_no_goal_nce`,
`R5_no_progress_rank`, `R6_no_action_rank`,
`R7_no_terminal_corrupt`, `R8_no_sigreg`,
`R9_no_temporal_straightening`.

Submitted training settings:

- optimizer steps: `60000`
- microbatch size: `8`
- gradient accumulation: `1`
- effective batch size: `8` full trajectories per optimizer step
- peak LR: `1e-4`
- warmup: `1000` steps
- schedule: linear warmup then cosine decay
- final LR: `1e-5`
- temporal straightening weight: `0.1`

Planner eval axes:

- Planner: MPC with beam search inner optimizer
- Beam width: `1,4,16,64`
- Beam depth: `8,16,32,64`
- Score: `oracle_goal_distance`, `predicted_goal_distance`
- Transition: `symbolic_reencode`, `latent_rollout`
