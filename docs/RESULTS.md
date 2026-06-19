# Results

Last updated: 2026-06-19 09:16 CEST

## Current Result

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

Final metric-sweep result: the strongest row is `R4_no_goal_nce` with
changed-cell raw Euclidean distance, beam width `8`, depths `16` and `32`,
which solved `3/4` boards (`solve_rate=0.75`) with mean remaining Hamming
`1.0`. `R1_no_context_masks` solved `1/4` (`0.25`) under several raw metrics,
including changed-cell raw, hybrid, cosine, raw squared, and progress/delta.
`M0_full` remained `0.0` across completed metric-sweep rows. The older
per-metric probes `3751931`-`3751938` timed out after 24h with 5/8 rows each;
best there remained `R1_no_context_masks` raw at `0.125`.

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
