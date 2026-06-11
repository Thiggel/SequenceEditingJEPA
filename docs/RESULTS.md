# Results

Last updated: 2026-06-11 09:56 CEST

Detailed results now live in `../sequence-editing-report/RESULTS.md` and the
ongoing LaTeX report `../sequence-editing-report/report.tex`.

## Current Key Result

## 2026-06-11 Local A100 Qualitative Probe

Script: `scripts/analysis/sudoku_hier_value_probe.py`.
Follow-up script: `scripts/analysis/sudoku_terminal_projection_probe.py`.

Artifacts:

- `/home/vault/c107fa/c107fa12/sequence-editing/analysis/sudoku_hier_value_terminal_local_20260611.json`
- `/home/vault/c107fa/c107fa12/sequence-editing/analysis/sudoku_hier_value_top_level_20260611.json`
- `/home/vault/c107fa/c107fa12/sequence-editing/analysis/sudoku_hier_value_mcts_root_20260611.json`
- `/home/vault/c107fa/c107fa12/sequence-editing/analysis/sudoku_hier_value_mcts_full_small_20260611.json`
- `/home/vault/c107fa/c107fa12/sequence-editing/analysis/sudoku_terminal_projection_probe_20260611.json`
- `/home/vault/c107fa/c107fa12/sequence-editing/analysis/sudoku_terminal_projection_smoke_20260611.json`

Main read: the learned value/energy heads can distinguish very wrong terminal
boards, but they are not safe at fine granularity and do not rank adjacent
actions reliably. On 32 eval boards, Grid 4M `state_value` picked the true
solution over corrupted terminal boards `88.5%` overall, but only `70.3%` for
one-cell corruptions. Grid 4M `terminal_energy` was stronger overall
(`96.7%`) but still only `87.5%` on one-cell corruptions. Local successor
ranking with `state_value` was poor: best goal-correct action top-1 was
`4/48 = 8.3%`, mean best-goal rank `6.58`.

Top-level hierarchy probes show the high-level optimizer can reduce predicted
latent goal MSE, but the generated first subgoal is not grounded for the
low-level CEM planner. Across two boards and CEM/GD/GD+reachability, the
low-level 16-action chunk averaged only `2.0` correct writes and `14.0` wrong
writes for Grid 4M `state_value`; Grid 4N macro-action advantage averaged
`2.42` correct and `13.58` wrong. Gradient optimization of learned scores can
also exploit off-manifold latents: e.g. `goal_value` GD pushed the raw value
head above `10` while worsening predicted latent MSE in some probes.

Terminal-depth MCTS does not solve the local-value issue at practical budgets.
With exact symbolic transitions, max depth set to the remaining blanks
(`54`), and `256/1024` simulations, the tree reached only depth `2-4`, found
`0` terminal nodes and `0` solved nodes, and learned `goal_value` root choices
were wrong in all sampled top actions. A tiny full MCTS/MPC check with 64
simulations per step solved `0/1`; learned `goal_value` ended with `50`
remaining Hamming and oracle `latent_goal` with `30`. The problem is not just
"score terminal states better"; the current tree almost never sees terminal
states and the nonterminal value is locally too flat/misordered.

The terminal-projection follow-up directly tested the stronger version of the
MCTS question: for every candidate first write, fill all remaining mutable
cells with the true solution, then score that terminal board. This is an oracle
terminal rollout upper bound, not a deployable planner. It still failed. Grid
4M `state_value` never ranked a true-solution terminal leaf first across 48
records; mean best solution-terminal rank was `156.38`, and within a fixed
cell the true digit was top-1 only `28.98%` of the time. Grid 4M
`terminal_energy` was better but still never ranked the true solution first
globally; mean best solution-terminal rank was `64.06`, and same-cell top-1 was
`59.45%`. Thus even if MCTS reached terminal leaves by a perfect completion, the
current learned terminal scorers would still often choose a one-cell-corrupt
terminal board.

The same follow-up checked whether the top hierarchy can at least find the
correct terminal latent. It cannot in the current checkpoints: nearest-terminal
analysis among the true solution plus all one-cell corruptions had true
terminal top-1 rate `0.0`; mean true-terminal rank was `178.75` for Grid 4M
`state_value` and `211.42` for Grid 4N macro-action advantage. Restricting
macro actions to a codebook of real encoded 16-step chunks, including the
oracle chunk, also failed: the oracle chunk was never top-1, with mean rank
`68.0` for `state_value` and `16.0` for macro-action advantage. The selected
codebook chunks and low-level CEM executions still contained mostly wrong
writes.

Verification: `python -m py_compile scripts/analysis/sudoku_hier_value_probe.py
scripts/analysis/sudoku_terminal_projection_probe.py` and
`pytest tests/test_puzzle_models.py -q` passed.

Grid 3D reset-large diagnostics confirm that periodic planner-state
re-encoding recovers the re-encoded oracle-goal result on a larger paired
sample. On 128 boards for the rollout `N=2` checkpoint, no-reset
terminal-energy planning solved `7/128`, reset every 4 actions solved
`128/128` under both step and terminal energy, reset every 8 solved `91/128`
under step energy but `128/128` under terminal energy, and full re-encoded
planning solved `128/128`.

| Run | Planner | Scoring | Solve | Terminal | Mean remaining Hamming |
| --- | --- | --- | ---: | ---: | ---: |
| lead large | latent rollout | terminal energy | 0.0 | 0.0625 | 4.671875 |
| rollout `N=2` | latent rollout | terminal energy | 0.0625 | 0.40625 | 2.453125 |
| Grid 3C paired | no reset | terminal energy | 0.03125 | 0.5625 | 2.265625 |
| Grid 3C paired | reset every 4 | step energy | 1.0 | 1.0 | 0.0 |
| Grid 3C paired | reset every 8 | terminal energy | 1.0 | 1.0 | 0.0 |
| Grid 3D paired | no reset | terminal energy | 0.0546875 | 0.3984375 | 2.3984375 |
| Grid 3D paired | reset every 4 | step energy | 1.0 | 1.0 | 0.0 |
| Grid 3D paired | reset every 8 | step energy | 0.7109375 | 0.7109375 | 0.2890625 |
| Grid 3D paired | reset every 8 | terminal energy | 1.0 | 1.0 | 0.0 |
| lead large | re-encoded state | terminal energy | 1.0 | 1.0 | 0.0 |
| rollout `N=2` | re-encoded state | terminal energy | 1.0 | 1.0 | 0.0 |

This passes the mechanism gate for a planner-state reset/re-encoding branch,
but it is still an oracle-goal diagnostic, not a deployable solver. The next
user-directed branch is Grid 4A: train one-, two-, and three-level JEPA models
with a learned goal-energy head, learned action-encoder hierarchy, and CEM.
Training array `3688587_[0-2]` started at `2026-06-01 13:06:00 CEST`, but it
was pre-HWM correction and was cancelled after user approval at `14:46:56 CEST`.
Intermediate corrected training `3688921_[0-2]` was cancelled at `15:01:20 CEST`
after the user requested the exact report-style high-level latent-action CEM
to subgoal and low-level primitive CEM recipe. Replacement training
`3688986_[0-2]` completed cleanly at step 5000 for all three levels. The
dependent learned-energy CEM diagnostics `3689396_[0-2]` and report-style
subgoal CEM diagnostics `3689397_[0-1]` also completed cleanly, but the actual
CEM solve gates failed: learned-energy CEM solved `0/64` for every level, and
subgoal CEM solved `0/32` for L2 and L3.

Grid 4B completed cleanly on 2026-06-02 and failed decisively: learned-energy
beam/reset solved `0/128` for L1/L2/L3. Paired reset mean remaining Hamming was
L1 `47.41`, L2 `46.23`, L3 `45.84`, with terminal rate `0.0`. This means the
learned goal-energy scorer is the immediate blocker, not just CEM.

Grid 4C `3695040` completed cleanly on 2026-06-03. It reused the L1 checkpoint
from Grid 4B but switched the reset/beam planner back to oracle solved-board
latent MSE. Reset every 4 and re-encoded oracle-goal planning solved `128/128`,
while no-reset latent planning solved `79/128` under terminal-energy selection.
This confirms the checkpoint dynamics are still compatible with the old
oracle-goal reset result; the failure is the learned scorer. Calibration records
show predicted energy follows the successful trajectories in aggregate but is
not reliable enough for local action selection.

Grid 4D `3696616_[0-5]` trained all six non-hierarchical L1 scorer variants and
completed all diagnostics. The result is still a hard fail for deployable
learned-energy planning: every variant solved `0/128`. Oracle-goal controls
show that margin and margin+mono preserve reset dynamics best: reset every 4
solved `128/128` for both, NCE solved `120/128`, and InfoNCE, NCE+mono, and
InfoNCE+mono solved `0/128`.

Grid 4E `3698281_[0-6]` completed cleanly. It confirms the local ranking
failure: original L1 gold top1 is `0.040`, and the six Grid 4D contrastive
variants range only `0.024-0.049`. Other-cell goal-correct actions outrank the
sampled gold action about half the time, so single-gold local negatives are the
wrong target for Sudoku.

Grid 4F completed. Unstratified CVL and MuZero-lite both solved `0/128` under
learned-energy reset/beam. MuZero-lite preserved the oracle-goal reset control
at `128/128`, while CVL did not.

Grid 4G stratified CVL completed as `3698893`; it solved `0/128` under
learned-energy reset/beam and also `0/128` under oracle-goal reset control.

Grid 4H `3698988` was cancelled because binary terminal correctness was too
sparse: it labeled solved boards as `1`, but reachable nonterminal boards as
`0`. Grid 4I `3699523` replaces it with discounted reachability: the scalar
head target is `0.99^N`, where `N` is remaining wrong-cell count to the
solution; impossible clue-corrupt states get target `0`.

Grid 4O `3714062_[0-3]` timed out after 8 hours. It was an inference-only
MCTS diagnostic on the existing original L1 checkpoint: symbolic transitions,
leaf re-encoding, learned `goal_energy` versus oracle `latent_goal`, depths
8/16, 512 simulations, and root-action debug records. No `diagnostics_mcts_*`
directories or MCTS JSON/JSONL records were written, so learned-vs-oracle MCTS
solve/terminal/Hamming and root-action metrics are unavailable. The next MCTS
run should stream per-example JSONL and/or use a smaller budget before scaling.

Grid 4P is the replacement smaller MCTS run after adding streaming/caching. It
completed as `3715249_[0-3]`, exit `0:0`, with 32 boards, 128 simulations,
expansion cap 32, and depths 4/8 for learned `goal_energy` and oracle
`latent_goal`. Learned `goal_energy` d4/d8 solved `0/32`, terminal `0`, and
mean remaining Hamming `47.78`/`48.72`; oracle `latent_goal` d4/d8 solved
`0/32`, terminal `0`, and mean remaining Hamming `9.88`/`10.03`. Root debug
shows learned `goal_energy` top-1 goal-value action rates `65/438` and
`70/452` versus oracle `latent_goal` `376/449` and `360/440`. Oracle MCTS is
directionally useful, but still does not solve at this budget. Mean Hamming
improvement is `7.28`/`7.25` for learned `goal_energy` d4/d8 versus
`46.09`/`45.53` for oracle `latent_goal`.
Grid 4I training completed, but the job hit `NODE_FAIL` before diagnostics.
Replacement diagnostics-only job `3702008` completed cleanly. Discounted
reachability solved `0/128` under learned-score reset/beam, with reset-every-4
mean remaining Hamming `55.40` and terminal rate `0.0`. The oracle latent-goal
control still solved `128/128` with reset every 4 and re-encoded planning, so
the dynamics were preserved; the learned value target is the failure.

Grid 4J `3702066` completed. It targets the original L1 terminal-distance head
and compares predicted scalar energy against true latent goal energy for all
candidate actions over 16 boards x 5 steps. Mean all-action absolute error is
small (`0.00443`), but mean within-step Pearson correlation is weak (`0.337`);
qualitative examples show wrong actions beating gold under predicted energy.

Grid 4K `3702254_[0-1]` completed cleanly. Both ListNet label variants still
solved `0/128` under learned-score reset/beam. Remaining-wrong relevance filled
boards but left mean remaining Hamming `47.72`; latent-goal relevance left
mean remaining Hamming `49.21`. Oracle controls separate the variants:
remaining-wrong relevance preserved reset-every-4 oracle planning at `128/128`,
while latent-goal relevance degraded it to `112/128`.

Grid 4L `3705899_[0-6]` completed normal diagnostics for all seven variants;
task 6 timed out only during the extra MCTS diagnostic. Every learned-score
reset/beam variant solved `0/128`; every oracle latent-goal reset control
solved `128/128`. Learned reset-every-4 remaining Hamming was: scaled energy
`45.78`, action advantage `53.93`, local z-score `48.83`, local margin `42.20`,
task value `49.39`, latent progress `46.91`, and MuZero-like value `49.86`.
Local z-score and local margin filled every board but still solved none, so
terminal fill is not enough.

Fixed-sign Grid 4I diagnostic `3705900` completed cleanly. Correct `goal_value`
sign improved terminal rate from `0.0` to `0.172` and mean remaining Hamming
from `55.40` to `49.83`, but solve stayed `0/128`.

Grid 4M `3711931_[0-3]` is running since 2026-06-10 11:42:19 CEST. Learned
flat reset still solves `0/128` for all four variants. Reset-every-4 mean
remaining Hamming is terminal energy `47.70`, action advantage `53.62`, state
value `49.73`, and contrastive margin `43.88`; contrastive margin fills nearly
all boards but still leaves wrong boards. Oracle latent-goal reset/calibration
now solves `128/128` with reset every 4 and re-encoded planning for all four
variants, so the dynamics/control remain intact. Partial latent-goal subgoal
CEM outputs now exist and are all poor: terminal energy solved `0/32`, terminal
`0/32`, mean remaining Hamming `49.44`; state value solved `0/32`, terminal
`0/32`, mean remaining Hamming `48.75`; contrastive margin solved `0/32`,
terminal `0/32`, mean remaining Hamming `49.59`; action advantage solved
`0/32`, terminal `0/32`, mean remaining Hamming `49.25`. Task `3711931_1`
completed cleanly; tasks `_0`, `_2`, and `_3` are still running. Learned
top-score subgoal directories for terminal energy, state value, and contrastive
margin exist but contain no final `diagnostics.json` yet.

Grid 4N `3711983` completed cleanly at 2026-06-11 06:45:29 CEST. Its oracle
latent-goal reset/calibration diagnostic passed: no-reset terminal-energy
selection solved `127/128` with terminal rate `1.0` and mean remaining Hamming
`0.008`, while reset every 4 and re-encoded planning solved `128/128`. The
subgoal reads are poor. Latent-goal subgoal CEM solved `0/32`, terminal
`0/32`, mean remaining Hamming `48.47`; learned macro-action top-score subgoal
CEM solved `0/32`, terminal `0/32`, mean remaining Hamming `49.72`. The
learned macro-action top score is not directionally useful in this diagnostic.

Grid 4Q/4R are queued recursive hierarchy diagnostics, not new training. They
add the report-style recursive planner: top-level CEM/GD/GD-with-reachability
optimizes latent macro-actions toward the global goal, the first predicted
latent becomes a subgoal for the next lower level, and primitive CEM only acts
at level 0. Grid 4Q depends on Grid 4M (`3715252_[0-11]` after `3711931`) and
remains dependency-blocked with no recursive artifacts because the Grid 4M
wrapper is still running post-training diagnostics. Grid 4R (`3715251_[0-2]`)
completed after Grid 4N. Recursive macro-action top score solved `0/16` and
terminal `0/16` for `cem`, `gd`, and `gd_reachability`; mean remaining Hamming
was `51.81`, `52.94`, and `51.31`. This is worse than the non-recursive
latent-goal subgoal CEM control, so the learned macro-action top score is not
directionally useful in the recursive planner either.

Five additional user-requested non-recurring oversight checks were submitted at
exact Europe/Berlin begin times. The first attempt `3715429`-`3715433` was
cancelled before start by stale watch `3715253`, which was then cancelled at
11:56:08 CEST. Replacement jobs `3715446`, `3715447`, `3715448`, and `3715449`
completed cleanly. The 08:00 check `3715450` started at 2026-06-11 08:00:26
CEST on `a1621` and confirmed `http_proxy`, `https_proxy`, `HTTP_PROXY`, and
`HTTPS_PROXY` inheritance in the live environment. It must not submit a
successor oversight job.

Literature note: MuZero/Dreamer/TD-MPC-style value heads are not the clean
non-RL target we need because they use reward, TD, or search labels. The closest
adjacent recipe is contrastive goal-conditioned reachability/value learning:
future or reachable states are positives and unrelated/wrong successors are
negatives. This points toward a multi-positive scorer objective rather than the
single-gold local-negative setup used in Grid 4D.

Clarification: the Grid 3C/3D result uses the filled solution board as an
oracle goal latent for planning diagnostics. It means reset every 4 can solve
`128/128` when the solved board is given as the goal state and the planner is
allowed to score candidate boards against that goal. It does not mean the model
can yet solve Sudoku without being given the solution or an external verifier.

Generated artifacts: `../sequence-editing-report/assets/grid3b/` contains the
lead and rollout `N=2` planning comparisons, drift curves, terminal
remaining-Hamming distributions, mismatch heatmaps, final training curve, CSV
tables, Grid 3C/Grid 3D reset-cadence plots/CSVs, and concrete paired examples.
`../sequence-editing-report/assets/grid4a/` contains the step-1000 exact-recipe
training summary CSV/PNG plus final training and CEM diagnostic summaries,
failure examples, and remaining-Hamming plot.

## Grid 3A Grounding Result

Grid 3A local value-only action injection finished and diagnostics confirmed
the main action-grounding result. Direct local injection strongly outperformed
the old global-broadcast action conditioning; both direct variants rank a
goal-correct action first on every sampled diagnostic state.

| Run | Step | Eval loss | Mean rank | H1/H2/H4 solve |
| --- | ---: | ---: | ---: | --- |
| `sudoku_jepa_5m_local_direct_uniform` | 5000 | 0.000187 | 15.96875 | 1.0 / 1.0 / 1.0 |
| `sudoku_jepa_5m_local_direct_weighted` | 5000 | 0.0000639 | 16.25 | 1.0 / 1.0 / 1.0 |
| `sudoku_jepa_5m_local_residual_weighted` | 5000 | 0.00234 | 115.96875 | 0.0 / 0.0 / 0.0 |
| `sudoku_jepa_5m_local_direct_changed_only` | 5000 | 0.11818 | 246.53125 | 0.0 / 0.0 / 0.0 |

All four roots have final `metrics.json`, `metrics.jsonl`, and `checkpoint.pt`.
The first dependent diagnostics array `3674779_[0-3]` failed because the wrapper
passed comma-separated `--horizons`; after a local fix and smoke test,
diagnostics were resubmitted as `3676904_[0-3]` and completed.

`H1/H2/H4 solve` is an online training metric over only 8 eval examples. It is
not the final solver metric: H1 scores legal one-step actions by predicted
next-latent distance to the goal latent; H2/H4 expand exact symbolic board
states for a short horizon and re-encode candidate terminal states. Treat
diagnostic terminal planning as the stricter Sudoku-solve read.

## Active Follow-Up

Grid 3B rollout `N=2` completed as `3680020` and diagnostics completed as
`3680021`. Final online metrics at step `5000` were eval loss `0.000138`,
oracle mean rank `12.34375`, and H1/H2/H4 solve `1.0 / 1.0 / 1.0`, but the
larger diagnostics show exact latent solve remains weak.

Grid 3C reset-cadence diagnostics completed as `3682924`; Grid 3D reset-large
confirmation completed as `3683903` and wrote
`diagnostics_reset_cadence_large/diagnostics.json` plus paired records. It
confirms reset every 4 as the cheapest exact cadence tested on the larger
sample. Oversight successor `3684889` hit `NODE_FAIL`; replacement oversight
`3687722` completed, oversight `3688542` completed, successors `3689344` and
`3689685` were cancelled before start, replacement `3691526` completed at
`2026-06-02 14:32:29 CEST`, and successor `3692215` was cancelled by user
request at `2026-06-02 14:40:41 CEST`. Recurring oversight is now disabled.
Grid 4A pre-correction training `3688587_[0-2]` was cancelled after preserving
step-1 metrics; intermediate `3688921_[0-2]` was cancelled after the exact
planner correction; replacement training completed as `3688986_[0-2]`. Final
training metrics look healthy: L1/L2/L3 eval loss `0.000118`/`0.000192`/`0.000156`,
goal-energy MSE `5.72e-05`/`0.000148`/`9.67e-05`, and online H1/H2/H4 solve
`1.0 / 1.0 / 1.0`. However, learned-energy CEM failed with solve `0/64`,
terminal rate `0.0`, and mean remaining Hamming `50.80`/`50.33`/`49.70`.
Report-style subgoal CEM also failed with solve `0/32`; L2 mean remaining
Hamming `48.31`, L3 mean remaining Hamming `49.28`, and only one L3 sample
was terminal, still wrong.

## Grid 3A Diagnostics

| Run | Goal-rank mean / top1 | Rank mean | Drift @10 / @20 / terminal | Terminal planning |
| --- | ---: | ---: | ---: | --- |
| `local_direct_uniform` | 1.0 / 1.0 | 18.42 | 0.119 / 1.788 / 2.060 | solve 0.0, terminal 0.0, remaining Hamming 5.625 |
| `local_direct_weighted` | 1.0 / 1.0 | 21.82 | 0.078 / 1.728 / 2.007 | solve 0.0, terminal 0.125, remaining Hamming 4.25 |
| `local_residual_weighted` | 2.085 / 0.493 | 124.20 | 2.761 / 103.3 / 1940 | solve 0.0, terminal 0.0, remaining Hamming 47.375 |
| `local_direct_changed_only` | 15.49 / 0.0566 | 242.79 | 1.818 / 1.864 / 1.883 | solve 0.0, terminal 0.0, remaining Hamming 54.375 |

Against Sudoku Grid 1 and Grid 2A, direct local injection greatly improves
action grounding and closed-loop proximity: Grid 1 mix50/50 had rank mean
`167.68` and remaining Hamming `51.75`, while Grid 2A `N=4` had rank mean
`209.14` and remaining Hamming `53.875`. Local direct weighted reduces the
closed-loop miss to about four cells, but long-horizon drift remains high
(`@20 1.728`, terminal `2.007`) and terminal solve is still `0.0`.

## Prior Read

- Grid 1 diagnostics showed true re-encoded oracle states were monotonic toward
  the goal, while predicted latent rollouts drifted badly.
- Grid 2A rollout training improved 10/20-step drift but worsened action rank
  and did not fix terminal planning.
- Local action injection fixes a major action grounding failure caused by global
  action broadcast. The remaining bottleneck is long-horizon drift /
  closed-loop exactness after locally grounded one-step predictions.
- The residual/delta variant is not currently a win. It predicts an additive
  correction to a contextual latent and accumulates errors under closed-loop
  rollout; its drift explodes by 20/terminal steps.
