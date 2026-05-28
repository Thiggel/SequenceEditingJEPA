# Results

Last updated: 2026-05-28

## Legacy Sequence-Editing Read

The old iGSM/LANO sequence-editing line is frozen in [`../legacy.md`](../legacy.md).
The short read is:

- causal LM remained the only strong iGSM model;
- x0 DLM was the strongest non-causal denoising baseline;
- JEPA/editing variants learned useful local signals but did not beat x0 DLM on
  answer accuracy;
- fully unrolled reasoning-trace JEPA was either too slow or had no answer signal.

## Puzzle Scaffold Validation

Current results are implementation and run checks:

| Check | Result |
| --- | --- |
| `python -m pytest -q tests` | Passed after cleanup. |
| `python -m compileall -q puzzle_jepa` | Passed before cleanup; no generated files are kept. |
| JEPA Sudoku smoke | Runs forward/backward on oracle transitions. |
| JEPA Maze smoke | Runs forward/backward on oracle transitions. |
| HRM Sudoku smoke | Runs supervised answer loss and Q-head loss. |
| TRM Sudoku smoke | Runs supervised answer loss and Q-head loss. |
| PTRM Sudoku smoke | Runs K stochastic TRM rollouts and Q-head selection. |
| Grid 0 local Sudoku one-step override | Ran with 5.26M trainable params; train loss `0.6904`, eval loss `0.6476`. |
| Grid 0 local Maze one-step override | Ran with 5.28M trainable params; train loss `0.7250`, eval loss `0.6726`. |
| Grid 0 Slurm | Completed as `3664581_[0-1]` with exit `0:0` on `a0833`; no OOMs. Sudoku final step `1000`: train loss `0.0142`, eval loss `0.0141`, oracle-action top1 `0.03125`, mean rank `30.56`, H=1 solve rate `0.0`. Maze final step `1000`: train loss `0.0168`, eval loss `0.0163`, oracle-action top1 `0.0`, mean rank `116.5`, H=1 solve rate `0.0`. |
| Grid 1 implementation checks | `python -m pytest -q tests` passed; `python -m compileall -q puzzle_jepa` passed; one-step local Grid 1 Sudoku and Maze mix70/30 smoke runs completed with metrics/checkpoints under `$PUZZLE_JEPA_WORK_ROOT/runs/_local_grid1_{sudoku,maze}_smoke`. |
| Grid 1 Slurm | Completed as `3665018_[0-4]` with exit `0:0` on `a0932`; no OOMs and all stderr files are empty. All five runs wrote `metrics.json`, `metrics.jsonl`, `checkpoint.pt`, and `checkpoint-5000.pt`. Final H=1/2/4 solve rates are all `0.0`. |
| Mutable Sudoku action space | Implemented clue-mask-aware Sudoku actions. Clues are immutable; non-clue cells can be overwritten and can temporarily violate constraints for mutable-world planning/training. |
| Grid 1 diagnostics | Preliminary array `3666870_[0-4]` completed with exit `0:0`. It exposed non-monotonic predicted latent goal energy, but used old fill-only Sudoku planning for rank/traces. Corrected diagnostics `3667044_[0-4]` completed with exit `0:0`; all five runs wrote `diagnostics.json` with embedded planner traces, `rank_records.jsonl`, `drift_records.jsonl`, and `latent_energy_mse.png`. |
| Diagnostics report | `docs/puzzle_jepa_diagnostics_report.tex` now summarizes Grid 1 results, corrected diagnostics, all five latent-energy figures, concrete drift records, and representative planner trace prefixes. Figure copies are in `docs/assets/puzzle-diagnostics/`. |
| Recurring oversight | Oversight `3669988` completed with exit `0:0` after submitting next oversight `3670421`, pending for 2026-05-28 12:12:52 CEST. |

## Grid 0 Interpretation

Grid 0 is an infrastructure pass, not a solver result. Both datasets trained,
evaluated, and checkpointed successfully with about 5.3M trainable parameters.
One-step latent prediction improved sharply from initialization:

| Run | Step 1 eval loss | Final eval loss | Step 1 mean rank | Final mean rank | Final H=1 solve |
| --- | ---: | ---: | ---: | ---: | ---: |
| `grid0_sudoku_jepa_5m_oracle_smoke` | `0.6319` | `0.0141` | `43.72` | `30.56` | `0.0` |
| `grid0_maze_jepa_5m_oracle_smoke` | `0.6421` | `0.0163` | `349.5` | `116.5` | `0.0` |

The useful signal is that dynamics learning works and Maze action ranking moves
well above initialization. The main issue is that raw latent distance is still
not enough for greedy H=1 planning; Sudoku action ranking remains especially
weak and noisy. Grid 1 should therefore focus on mutable off-trajectory dynamics
and H=1/2/4 planning diagnostics rather than claiming solve accuracy from this
smoke grid.

## Grid 1 Results

Grid 1 now tests the documented curriculum: Sudoku oracle, Sudoku 70/30,
Sudoku 50/50, Maze oracle, and Maze 70/30. The mixed-data sampler keeps Sudoku
clues immutable while allowing random mutable-cell digits, conflicts, and
overwrites; Maze random transitions mark arbitrary original empty cells as path
tokens. Evaluation now emits H=1/2/4 planning metrics plus oracle/random
latent-distance deltas.

The array `3665018_[0-4]` completed on 2026-05-26 with exit `0:0` for every
task. Final output roots are:

- `$PUZZLE_JEPA_WORK_ROOT/runs/sudoku_jepa_5m_oracle`
- `$PUZZLE_JEPA_WORK_ROOT/runs/sudoku_jepa_5m_mix70_30`
- `$PUZZLE_JEPA_WORK_ROOT/runs/sudoku_jepa_5m_mix50_50`
- `$PUZZLE_JEPA_WORK_ROOT/runs/maze_jepa_5m_oracle`
- `$PUZZLE_JEPA_WORK_ROOT/runs/maze_jepa_5m_mix70_30`

All runs have final `metrics.json` plus `checkpoint.pt` and
`checkpoint-5000.pt`.

| Run | Train loss | Eval loss | Top1 | Mean rank | H1 solve | H2 solve | H4 solve | Oracle delta | Random delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sudoku_jepa_5m_oracle` | `0.0084` | `0.0085` | `0.0625` | `36.50` | `0.0` | `0.0` | `0.0` | `0.0084` | `-0.0008` |
| `sudoku_jepa_5m_mix70_30` | `0.0090` | `0.0084` | `0.03125` | `28.16` | `0.0` | `0.0` | `0.0` | `0.0087` | `-0.0022` |
| `sudoku_jepa_5m_mix50_50` | `0.0082` | `0.0074` | `0.03125` | `49.03` | `0.0` | `0.0` | `0.0` | `0.0079` | `-0.0001` |
| `maze_jepa_5m_oracle` | `0.0020` | `0.0021` | `0.0` | `16.00` | `0.0` | `0.0` | `0.0` | `0.0014` | `-0.0003` |
| `maze_jepa_5m_mix70_30` | `0.0022` | `0.0019` | `0.0` | `245.50` | `0.0` | `0.0` | `0.0` | `0.0009` | `-0.0005` |

Interpretation: Grid 1 is a clean systems pass but not yet a planning success.
Lower one-step prediction loss does not imply useful greedy latent-distance
planning. The Sudoku 70/30 mix gives the best final oracle-action mean rank, but
the 50/50 mix has worse ranking despite the lowest eval loss. Maze oracle is
much better ranked than Maze 70/30, so off-path random marks are not helping the
current scorer. The next safe step is planner/scorer diagnostics or value/energy
shaping, not a capacity sweep.

## Grid 1 Diagnostics Results

Diagnostics arrays `3666870_[0-4]` and corrected `3667044_[0-4]` evaluated the
completed Grid 1 checkpoints without training new weights. The corrected array
completed on 2026-05-27 with exit `0:0` for all tasks. It used clue-mask mutable
Sudoku actions, latent unrolls through `predict_latent_from_latent`, bounded
step-energy and terminal-energy beam planning, and saved latent-energy plots.
All final diagnostics stderr files are empty.

Preliminary `3666870` result, before the mutable Sudoku action-space correction:

- Predicted latent goal energy under oracle action unrolls is not monotonic:
  Sudoku monotone rates were about `0.29-0.35`, Maze about `0.37-0.44`.
- True re-encoded oracle states have nearly monotonic goal energy: about `1.0`
  for all runs.
- This supports the concern that intermediate energy is unreliable. Terminal
  LeWM-style scoring is the right next diagnostic before temporal straightening
  or value shaping.
- Sudoku rank/traces from `3666870` are superseded because they used fill-only
  actions. Corrected job `3667044` overwrote the diagnostics outputs using
  clue-mask mutable actions.

Outputs per run:

- `diagnostics/diagnostics.json`: aggregate rank, drift, planning, and embedded
  planner traces.
- `diagnostics/rank_records.jsonl`: oracle-action rank by trajectory depth.
- `diagnostics/drift_records.jsonl`: latent drift and goal-energy records at
  steps `1,2,4,10,20` plus terminal or max-unroll step.
- `diagnostics/latent_energy_mse.png`: plot of predicted-goal MSE, true
  encoded-state goal MSE, and latent drift MSE over oracle unroll steps.

Questions this answers:

1. Does the world model compose under oracle actions, or does latent drift grow
   quickly after a few steps?
2. Is latent energy to the oracle goal monotonic along true oracle trajectories,
   or is it non-monotonic enough that intermediate greedy scoring is unreliable?
3. Is terminal latent energy low after a full oracle unroll, which is the
   closest current diagnostic to LeWM-style terminal scoring?
4. Does oracle-action rank degrade with trajectory depth?
5. Do short H=1 failure traces show wrong-action lock-in, tiny score margins, or
   an action-space mismatch?
6. Does bounded terminal LeWM-style beam planning behave differently when scored
   at every step versus when terminal states are reranked by final latent energy?

Corrected diagnostics from `3667044_[0-4]`:

| Run | Rank top1 | Rank top5 | Mean rank | Predicted energy monotone | True energy monotone | Oracle-unroll terminal predicted MSE | Step/terminal beam solve |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `sudoku_jepa_5m_oracle` | `0.0088` | `0.0303` | `184.79` | `0.334` | `1.000` | `1.753` | `0.0`/`0.0` |
| `sudoku_jepa_5m_mix70_30` | `0.0127` | `0.0420` | `182.01` | `0.352` | `0.999` | `1.729` | `0.0`/`0.0` |
| `sudoku_jepa_5m_mix50_50` | `0.0127` | `0.0361` | `167.68` | `0.287` | `1.000` | `1.710` | `0.0`/`0.0` |
| `maze_jepa_5m_oracle` | `0.0078` | `0.0781` | `124.12` | `0.372` | `1.000` | `1.938` | `0.0`/`0.0` |
| `maze_jepa_5m_mix70_30` | `0.0195` | `0.0391` | `205.27` | `0.436` | `1.000` | `1.956` | `0.0`/`0.0` |

The key read is that expanding to the correct mutable action space makes local
action ranking much harder, especially early in trajectories. The true state
encoding has the expected monotonic goal-energy behavior along oracle
trajectories, but the predicted latent rollout does not: by true oracle terminal
states, predicted latents are still about `1.7-2.0` MSE from the goal. Terminal
LeWM-style reranking does not help yet because bounded beam planning reaches no
terminal Sudoku boards or Maze paths. The concrete bottleneck is latent rollout
drift plus weak early action ranking, not capacity.
