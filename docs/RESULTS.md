# Results

Last updated: 2026-07-01 20:17 CEST

## Current H1 Recipe / Old-Local Fast Wave

Minimal-aux 5k single-factor wave is now the active source of results. Train
array `3803494` completed all 29 variants cleanly. Eval array `3803495` is
running all 29 tasks and has written `141 / 456` expected rows so far. Current
rows are `mpc_beam` only; `hierarchical_beam` rows have not appeared yet.

Best partial rows:

| Variant | Goal/score | Depth | Result |
| --- | --- | ---: | --- |
| `goal_distance_field_distill` | oracle raw L2 | 4 | `8/8`, h `0.0` |
| `reg_sigreg` | oracle normalized | 4 | `8/8`, h `0.0` |
| `base` | oracle normalized | 4 | `7/8`, h `0.125` |
| `hier_l4_l16` | oracle raw L2 | 4 | `7/8`, h `0.125` |
| `reg_vicreg` | predicted normalized | 4 | `0/8`, h `34.6` |

Interpretation: the 5k minimal-aux recipe already recovers strong oracle
global latent-rollout planning. Predicted-goal planning remains the bottleneck:
all partial predicted-goal rows are still `0/8`.

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
