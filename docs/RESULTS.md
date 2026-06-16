# Results

Last updated: 2026-06-16 18:22 CEST

## Current Result

No Grid-Token Goal-JEPA jobs have been submitted yet.

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
- Training now uses microbatch `64`, gradient accumulation `4`, effective
  batch size `256`.
- Current verification: `source scripts/env.sh && pytest -q` -> `31 passed`.
- Additional verification: `source scripts/env.sh && python -m compileall -q
  puzzle_jepa configs` and Slurm launcher syntax checks passed.
- Running `pytest -q` without `source scripts/env.sh` fails at collection
  because the default Python cannot import `torch`.

No Grid-Token jobs have been submitted. Submit only after the user says `go`.

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

Full experiment suite is now submitted on RTX Pro 6000:

- Training array `3748789`: 13 ablations, 60k optimizer steps, batch 8, no
  grad accumulation
- Dependency-held planner eval array `3748790`: pending on successful
  completion of all training array tasks

## Legacy Result

The previous faithful LeWM/CLS/value-head reset is now legacy. Its main result
was negative for Sudoku planning geometry: exact symbolic and true-Hamming
oracle scoring could solve, but oracle latent distance and learned scalar
goal-distance scoring did not produce solves. That result motivated the current
full-grid goal-prediction architecture.
