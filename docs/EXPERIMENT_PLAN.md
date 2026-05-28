# Experiment Plan

Last updated: 2026-05-28

## Pivot

The project has pivoted from denoising free-form reasoning traces to objective
puzzle worlds: Sudoku-Extreme and Maze-Hard first, with ARC left for later. The
legacy sequence-editing code is archived at `../legacy-sequence-editing` and the
method/results summary is in [`../legacy.md`](../legacy.md).

## Paper And Code Read

| Line | Source | Implementation read |
| --- | --- | --- |
| HRM | arXiv `2506.21734`, `https://github.com/sapientinc/HRM` | Two recurrent modules (`z_H`, `z_L`), deep supervision, detached recurrent carry between segments, ACT/Q head, dataset builders for `sapientinc/sudoku-extreme` and `sapientinc/maze-30x30-hard-1k`. |
| TRM | arXiv `2510.04871`, `https://github.com/SamsungSAILMontreal/TinyRecursiveModels` | One tiny recurrent network updates latent `z` and answer `y`; full local recursion; EMA; simplified Q/halting head. TRM-MLP is strongest for Sudoku-Extreme, TRM-Att for Maze-Hard/ARC. |
| PTRM | arXiv `2605.19943`, `https://amins01.github.io/ptrm/` | Code was not available from the project page. Method is inference-only: inject Gaussian noise into TRM latent recursions, run K rollouts, select by Q head or mode. |

## JEPA Training Plan

Train the JEPA world model on objective transitions, not text traces.

For Sudoku:

1. Sample a puzzle and solution.
2. Sample a valid partial board by keeping clues and a random subset of solution
   cells.
3. Pick a mutable cell. Clue cells are immutable; non-clue cells may be empty or
   overwritten.
4. Use action `(row, col, digit)`.
5. Target the next board with that mutable cell set to `digit`.

For Maze:

1. Sample a maze and oracle shortest path.
2. Sample a partial path state by revealing a subset of oracle path cells.
3. Pick a remaining path cell.
4. Use action `(row, col, PATH)`.
5. Target the next maze with that path cell marked.

Training should be staged by question:

1. **Oracle dynamics.** Train on valid oracle-improving transitions only. This
   answers whether the architecture can learn clean objective state dynamics and
   whether latent planning works when all rollouts stay on the solution manifold.
2. **Full mutable dynamics.** Add random valid actions on mutable cells,
   including actions that create row/column/block conflicts or overwrite a prior
   model-written cell. These are real environment transitions, not invalid
   transitions. This answers whether the world model predicts off-trajectory
   states rather than only memorizing solution prefixes.
3. **Preference/energy shaping.** Add a value, verifier, or contrastive latent
   energy objective so states closer to an oracle goal score better than
   constraint-violating or farther states. This answers whether latent distance
   or a learned scalar can guide search, rather than only predict next states.
4. **Hard invalid actions.** Keep clue-cell overwrites and out-of-range actions
   outside inference by action masking. If used in training, use them only as
   explicit negative validity examples because they have no environment
   transition.

The main distinction is that constraint-violating Sudoku boards can still be
valid *states* under a mutable-cell environment, while clue overwrites are not
valid *actions*. Pure dynamics training can include the former; search guidance
still needs latent/energy shaping because next-state prediction alone does not
guarantee that Euclidean latent distance is monotonic with distance to the goal.

## First Run Matrix

### Grid 0: Local/Single-GPU Pipeline Proof

Submitted on 2026-05-26 as Slurm array `3664581_[0-1]` with recurring oversight
`3664583`. Completed successfully with exit `0:0` at `2026-05-26 16:23:58 CEST`.
It is not for a paper number; it proves the train/eval loop, checkpointing, and
metric schema.

| Run | Model | Data mix | Train budget | Batch | LR | Eval |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `grid0_sudoku_jepa_5m_oracle_smoke` | JEPA, 5.26M trainable params | 100% oracle mutable transitions | `1000` steps | `1024` | `1e-4` | one-step latent MSE, oracle-action rank, greedy H=1 planning on 8 puzzles |
| `grid0_maze_jepa_5m_oracle_smoke` | attention JEPA, 5.28M trainable params | 100% oracle path transitions | `1000` steps | `16` | `1e-4` | one-step latent MSE, oracle-action rank, greedy H=1 planning on 1 maze |

Result: no OOMs, final checkpoints exist, and one-step prediction improved from
initialization on both datasets. H=1 greedy planning still had `0.0` solve rate,
so this is only a pipeline pass. Maze action ranking improved substantially;
Sudoku action ranking remains weak/noisy.

### Grid 1: Core Data Curriculum

This is the first real grid. Keep size fixed at about 5M and test the central
question: does off-policy mutable dynamics help planning or just make the latent
space harder?

| Run | Model | Data mix | Batch | LR | Main question |
| --- | --- | --- | ---: | ---: | --- |
| `sudoku_jepa_5m_oracle` | JEPA | 100% oracle transitions | `1024-2048` | `1e-4` | Clean dynamics baseline. |
| `sudoku_jepa_5m_mix70_30` | JEPA | 70% oracle, 30% random mutable actions | `1024-2048` | `1e-4` | Does off-trajectory coverage improve recovery/search? |
| `sudoku_jepa_5m_mix50_50` | JEPA | 50% oracle, 50% random mutable actions | `1024-2048` | `1e-4` | How much random dynamics can we add before target-trajectory quality drops? |
| `maze_jepa_5m_oracle` | attention JEPA | 100% oracle path transitions | `128-256` | `1e-4` | Clean maze dynamics baseline. |
| `maze_jepa_5m_mix70_30` | attention JEPA | 70% oracle, 30% random mutable path/non-path marks if enabled | `128-256` | `1e-4` | Does off-path coverage improve planning? |

Evaluate H=1/2/4 fixed-horizon planning with oracle goal latents. Also report
latent distance monotonicity along oracle trajectories and after random bad
actions.

Implemented, submitted, and completed on 2026-05-26 as Slurm array
`3665018_[0-4]`; all five tasks ran on `a0932` from
`2026-05-26 20:29:34 CEST` and exited `0:0`. Configs:
`grid1_sudoku_jepa_5m_oracle`, `grid1_sudoku_jepa_5m_mix70_30`,
`grid1_sudoku_jepa_5m_mix50_50`, `grid1_maze_jepa_5m_oracle`, and
`grid1_maze_jepa_5m_mix70_30`.

Implementation notes:

- Sudoku random mutable transitions keep clue cells fixed, allow any digit in
  mutable cells, and allow overwrites of prior mutable-cell values.
- Maze random mutable transitions mark arbitrary original empty cells as `PATH`,
  so the 70/30 mix includes off-path marks.
- Evaluation now logs `planning_h1`, `planning_h2`, and `planning_h4` solve
  rates/mean steps plus `oracle_latent_delta` and `random_latent_delta`.
- Submitted configs use `5000` steps. Sudoku batch is `1024`. Maze batch is
  conservatively `16` because Grid 0 already used about `22.3 GiB` at batch
  `16`; the original `128-256` target is deferred until memory behavior is
  optimized or gradient accumulation is added.

Final result:

| Run | Eval loss | Top1 | Mean rank | H1/H2/H4 solve | Oracle delta | Random delta |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| `sudoku_jepa_5m_oracle` | `0.0085` | `0.0625` | `36.50` | `0.0`/`0.0`/`0.0` | `0.0084` | `-0.0008` |
| `sudoku_jepa_5m_mix70_30` | `0.0084` | `0.03125` | `28.16` | `0.0`/`0.0`/`0.0` | `0.0087` | `-0.0022` |
| `sudoku_jepa_5m_mix50_50` | `0.0074` | `0.03125` | `49.03` | `0.0`/`0.0`/`0.0` | `0.0079` | `-0.0001` |
| `maze_jepa_5m_oracle` | `0.0021` | `0.0` | `16.00` | `0.0`/`0.0`/`0.0` | `0.0014` | `-0.0003` |
| `maze_jepa_5m_mix70_30` | `0.0019` | `0.0` | `245.50` | `0.0`/`0.0`/`0.0` | `0.0009` | `-0.0005` |

Grid 1 is a clean infrastructure pass but does not identify a solver-ready data
mix. The best Sudoku rank is the 70/30 mix, but the 50/50 mix has lower eval
loss and worse ranking. Maze oracle is much better ranked than Maze 70/30.
Capacity sweeps should wait until planner/scorer diagnostics produce a
measurable planning signal.

### Grid 2: Size Ablation

Run this only after a follow-up diagnostic identifies a data mix or scorer with
measurable planning signal. Do not sweep LR yet unless the chosen setup is
unstable.

| Run | Model | Data mix | Batch | LR | Main question |
| --- | --- | --- | ---: | ---: | --- |
| `sudoku_jepa_5m_bestmix` | JEPA | best Grid-1 mix | `1024-2048` | `1e-4` | Small baseline. |
| `sudoku_jepa_10m_bestmix` | JEPA | best Grid-1 mix | `512-1024` | `1e-4` | Does capacity improve latent planning? |
| `sudoku_jepa_20m_bestmix` | JEPA | best Grid-1 mix | `256-512` | `1e-4` | Does performance saturate or overfit? |
| `maze_jepa_5m_bestmix` | attention JEPA | best Grid-1 mix | `128-256` | `1e-4` | Small maze baseline. |
| `maze_jepa_10m_bestmix` | attention JEPA | best Grid-1 mix | `64-128` | `1e-4` | Capacity test under 900-token attention. |
| `maze_jepa_20m_bestmix` | attention JEPA | best Grid-1 mix | `32-64` | `1e-4` | Upper bound before larger compute. |

### Grid 3: Search/Test-Time Compute

Run on the best checkpoints from Grid 2. This grid may be eval-only.

| Eval | Horizon | Branching | Scorer | Main question |
| --- | ---: | ---: | --- | --- |
| `horizon_sweep_latent_distance` | `1,2,4,8,16` | top `8-64` actions | distance to oracle goal latent | Does more test-time compute improve solve rate? |
| `horizon_sweep_reencode` | `1,2,4,8,16` | top `8-64` actions | re-encode symbolic next states | Is latent-only rollout error the bottleneck? |
| `oracle_action_rank_curve` | n/a | all legal actions | oracle action rank | Does the model know the correct next move locally? |
| `bad_action_distance_curve` | n/a | sampled random mutable actions | latent distance and true Hamming distance | Are bad states farther from the goal in representation space? |

### Grid 4: Energy/Verifier Shaping

Only run this if Grid 3 shows that raw latent distance is not monotonic enough
for search.

| Run | Base | Extra objective | Negatives | Main question |
| --- | --- | --- | --- | --- |
| `sudoku_jepa_energyrank_5m` | best 5M JEPA | margin rank: oracle next closer than random next | random mutable actions, conflict states | Can headless latent distance be shaped enough? |
| `sudoku_jepa_value_5m` | best 5M JEPA | scalar value/energy head `V(state, goal)` | random mutable actions, conflict states | Does a learned scorer beat raw latent distance? |
| `maze_jepa_value_5m` | best maze 5M JEPA | scalar value/energy head | off-path partial paths | Is Maze limited by verifier/scorer quality? |

### Grid 5: Recursive Baselines

Run in parallel with Grid 2 or after Grid 2 if compute is tight. These reproduce
the relevant baselines for the same data/eval split.

| Run | Model | Size | Dataset | Main metric |
| --- | --- | ---: | --- | --- |
| `sudoku_hrm_repro` | HRM | paper-like, about 27M if feasible | Sudoku-Extreme | exact solve |
| `sudoku_trm_mlp_repro` | TRM-MLP | about 5M | Sudoku-Extreme | exact solve |
| `sudoku_ptrm_repro` | PTRM over TRM | about 5M, K sweep | Sudoku-Extreme | best-Q@K / mode@K |
| `maze_hrm_repro` | HRM | paper-like | Maze-Hard | exact path |
| `maze_trm_att_repro` | TRM-Att | about 7M | Maze-Hard | exact path |
| `maze_ptrm_repro` | PTRM over TRM-Att | about 7M, K sweep | Maze-Hard | best-Q@K / mode@K |

Use AdamW, bf16, gradient clipping `1.0`, target-encoder EMA `0.99-0.995`, and
weight decay `0.05-0.1` for JEPA. Start with `1e-4` and sweep `3e-4`/`3e-5`
only after the data/eval path is stable.

## Evaluation

Primary JEPA evaluation:

1. Encode the oracle goal state.
2. Enumerate all legal symbolic actions from the current state.
3. Predict the next latent for each `(state, action)`.
4. Score actions by distance to the oracle goal latent.
5. Apply the best action symbolically and repeat.

Report:

- exact solve rate;
- per-step oracle-action rank;
- invalid-action rate if invalid proposals are ever allowed;
- planning depth until solved;
- latent distance curves for solved versus failed examples.

Also run fixed-horizon planning sweeps. For horizon `H=1..N`, unroll the world
model under candidate action sequences, score the final predicted latent against
the oracle goal latent or value head, and apply the best first action. This tests
whether test-time compute improves solve rate and whether latent/energy distance
is monotonic along successful trajectories.

## Next Implementation Work

Current operational status as of 2026-05-28 09:55 CEST: no puzzle training or
diagnostics jobs are active. Grid 0 and Grid 1 remain complete with final
metrics and checkpoints. Grid 1 preliminary diagnostics `3666870_[0-4]`
completed and found non-monotonic predicted goal energy under oracle latent
unrolls. Corrected diagnostics `3667044_[0-4]` completed with clue-mask mutable
Sudoku actions, terminal LeWM-style beam planning, and latent-energy plots.
Oversight `3669988` completed cleanly after submitting recurring oversight
`3670421`, pending for 2026-05-28 12:12:52 CEST.
The Grid 1 diagnostic report is available as
`docs/puzzle_jepa_diagnostics_report.tex` with local figure assets in
`docs/assets/puzzle-diagnostics/`.

Corrected diagnostics identify two concrete bottlenecks:

- Predicted latent rollouts drift badly. True re-encoded oracle trajectories
  have monotonic goal energy (`0.999-1.000` monotone rate), but predicted latent
  rollouts are non-monotonic (`0.287-0.436` monotone rate) and still sit about
  `1.7-2.0` MSE from the goal at true oracle terminal states.
- Local action ranking is weak in the full mutable action space. Sudoku top1 is
  `0.0088-0.0127`; Maze top1 is `0.0078-0.0195`. Rank improves late in
  trajectories but early decisions are poor, and bounded step-energy and
  terminal-energy beam planning reach no terminal boards or paths.

Do not submit Grid 2 capacity ablations until at least one 5M checkpoint has a
nonzero final planning signal, or until a concrete planner/evaluation bug is
fixed.

1. Run the next eval-only diagnostic as a targeted Grid 3 slice: compare
   latent-only rollouts with re-encoding symbolic candidate states after each
   step. If re-encoding recovers rank/planning, rollout drift is the blocker; if
   it does not, the scorer/action-ranking objective is the blocker.
2. Compare step-energy beam planning against terminal-energy reranking only
   after the beam can reach terminal states. Current terminal reranking has no
   opportunity to help because terminal rate is `0.0`.
3. For the next training grid, remove task-id conditioning and the selected-cell
   marker because this project trains one model per task and action `(row, col,
   value)` already identifies the changed cell. Do not evaluate old checkpoints
   with those pieces removed because they were trained with them active.
4. Inspect why H=4 planning had a transient `0.125` solve rate at Grid 1 step 1
   for Sudoku oracle but `0.0` at later checkpoints; this may be small-sample
   noise, tie behavior, or a scorer regression.
5. Add value/verifier heads if re-encoded rollout diagnostics confirm that raw
   oracle-goal latent distance is the bottleneck rather than a rollout bug.
6. Add baseline reproduction configs for HRM, TRM, and PTRM.
