# Experiment Plan

Last updated: 2026-06-12 09:57 CEST

The active backlog now lives in `../sequence-editing-report/BACKLOG.md`.
Deferred planner-ablation notes live in `docs/PLANNER_ABLATION_NOTES.md`.

## Active Experiments

Grid 3B Sudoku follow-up:

| Run | Purpose | Status |
| --- | --- | --- |
| `sudoku_jepa_5m_local_direct_weighted` large diagnostics | Increase eval sample size and compare latent rollout planning with re-encoded symbolic-state planning; write terminal board records. | Completed as `3680019`; re-encoded planning solved `64/64`, latent rollout solved `0/64`. |
| `sudoku_jepa_5m_local_direct_weighted_rollout_n2` | Train direct local weighted JEPA with rollout loss `N=2`. | Completed as `3680020`; final step `5000`, eval loss `0.000138`, online H1/H2/H4 solve `1.0 / 1.0 / 1.0`. |
| Grid 3B rollout `N=2` diagnostics | Same larger diagnostics after rollout training. | Completed as `3680021`; latent terminal-energy solve `4/64`, re-encoded planning `64/64`. |
| Grid 3C reset/re-encoding diagnostic | Test periodic candidate-state re-encoding or latent reset cadence before broad scaling. | Completed as `3682924`; reset every 2/4 solved `64/64` paired boards under step and terminal energy, while no-reset terminal energy solved `2/64`. |
| Grid 3D reset-large confirmation | Confirm the reset/re-encoding branch on a larger paired sample before changing planner defaults or scaling. | Completed as `3683903`; reset every 4 solved `128/128`, reset every 8 solved `128/128` only under terminal-energy selection. |
| Grid 4A goal-energy / hierarchy / CEM | Train one-, two-, and three-level JEPA variants with a learned goal-energy head and evaluate with categorical CEM plus exact report-style hierarchical subgoal CEM. | Completed: training `3688986_[0-2]`, learned-energy CEM `3689396_[0-2]`, and subgoal CEM `3689397_[0-1]` all exited `0:0`, but CEM solve rate was `0.0` across the grid. |
| Grid 4B learned-energy reset beam | Test beam search with symbolic board state, learned goal-energy scoring, and reset/re-encode cadence 4 on the Grid 4A checkpoints. | Completed as `3691590_[0-2]`, exit `0:0`; learned-energy beam/reset solved `0/128` for L1/L2/L3. |
| Grid 4C L1 oracle reset/calibration sanity | Reuse the exact L1 checkpoint from `3691590_0`, switch planning back to oracle solved-board latent MSE (`--planning-score latent_goal`), and write learned-energy-vs-true-distance trajectory calibration plots. | Completed as `3695040`, exit `0:0`; reset every 4 and re-encoded oracle-goal planning solved `128/128`. |
| Grid 4D L1 contrastive goal-energy losses | Non-hierarchical L1 JEPA with existing goal-energy regression plus local successor negatives: `{nce,infonce,margin}` crossed with monotonicity off/on. | Completed as `3696616_[0-5]`; learned-energy reset/beam solved `0/128` for all variants. Oracle-goal reset controls recovered for margin and margin+mono, but not for InfoNCE-family variants. |
| Grid 4E action-candidate rank analysis | For sampled oracle trajectories, compare the gold action at each step against all alternative mutable-cell/value actions under learned goal-energy scoring, grouped into same-cell wrong value, other-cell goal value, and other-cell wrong value. | Completed as `3698281_[0-6]`; original L1 top1 `0.040`, best contrastive top1 `0.049`, so current losses still fail local action ranking. |
| Grid 4F value-method ablations | Test two literature-inspired scorer objectives on non-hierarchical L1: CVL multi-positive InfoNCE and MuZero-lite policy/value shaping. | Completed. Both solved `0/128` under learned-energy reset/beam; MuZero-lite preserved oracle-goal reset control, CVL did not. |
| Grid 4G stratified CVL scorer | Same CVL objective as Grid 4F, but the auxiliary batch is structured as multiple states per puzzle: `16` puzzles x `4` states/puzzle. | Completed as `3698893`; solved `0/128` under learned-energy and oracle-goal reset controls. |
| Grid 4H terminal-correctness scorer | Replace scalar latent-energy regression with a direct balanced terminal-correctness target on the existing scalar head. | Cancelled as `3698988`; sparse target was wrong for reachable nonterminal boards. |
| Grid 4I discounted reachability scorer | Corrected value target: scalar head predicts `0.99^N` for `N` remaining wrong cells, and `0` for impossible clue-corrupt states. | Completed via replacement diagnostics `3702008`; learned-score reset/beam solved `0/128`, while oracle reset every 4 solved `128/128`. |
| Grid 4J original L1 energy-action calibration | Qualitative and aggregate diagnostic comparing predicted scalar energy to true latent goal energy over all candidate actions. | Completed as `3702066`; small absolute errors but weak local correlation, with qualitative wrong-action wins. |
| Grid 4K ListNet learned-energy ranking | Train the existing L1 scalar head with listwise action-candidate ranking. Array task 0 uses discounted remaining-wrong-cell relevance `0.99^N`; task 1 uses true terminal latent goal-distance relevance. | Completed as `3702254_[0-1]`; learned-score reset/beam solved `0/128` for both. Oracle reset control solved `128/128` for remaining-wrong and `112/128` for latent-goal. |
| Grid 4L scorer-spread L1 ablation | Seven non-hierarchical scorer variants: scaled terminal energy, action advantage, local z-scored regression, local margin ranking, task-unit discounted value, latent progress shaping, and MuZero-like value+MCTS without policy head. | Completed normal diagnostics as `3705899_[0-6]`; task 6 timed out only during extra MCTS. Every learned scorer solved `0/128`; every oracle reset control solved `128/128`. |
| Grid 4I fixed-sign value diagnostic | Reuse existing discounted reachability checkpoint, but evaluate with `--planning-score goal_value` so higher value is selected. | Completed as `3705900`; correct sign improved terminal rate/remaining Hamming but solve stayed `0/128`. |
| Grid 4M hierarchical value L3 span-4 | Three-level hierarchy with `hierarchy_span=4`: terminal energy, primitive action advantage, discounted state value, and contrastive-margin energy. Each run evaluates flat learned-score reset, flat oracle reset, and level-2 oracle subgoal CEM; the three state-scorer variants also evaluate learned top-level subgoal CEM. | Mixed completion: task `_0` timed out after `1-00:00:26`; `_1`-`_3` completed cleanly. Useful reset/oracle diagnostics exist. Learned-score reset solves `0/128` for every variant; oracle latent-goal reset/calibration solves `128/128` for all four. The timeout makes Grid 4Q's `afterok:3711931` dependency unsatisfiable. |
| Grid 4N macro-action advantage L3 span-4 | True macro-action advantage head on `(initial latent, current latent, continuous level-2 macro action)`, trained on oracle 16-step chunks and used as top-level subgoal CEM score. | Completed as `3711983`, exit `0:0`. Oracle latent-goal reset/calibration passed (`127/128` no-reset terminal-energy, `128/128` reset every 4). Latent-goal subgoal CEM solved `0/32`, mean remaining Hamming `48.47`; learned macro-action top-score subgoal CEM solved `0/32`, mean remaining Hamming `49.72`, so the learned top score does not look directionally useful in this diagnostic. |
| Grid 4O inference-only MCTS value diagnostic | Existing original L1 goal-energy checkpoint, no training. MCTS uses exact symbolic Sudoku transitions, re-encodes leaf boards, and scores leaves with learned `goal_energy` or oracle `latent_goal`. Depths 8/16 were intended to test whether deeper tree search can rescue a locally flat scorer. | `3714062_[0-3]` timed out after 8h with no MCTS artifacts. Next run should stream per-example JSONL and/or use a smaller budget before scaling. |
| Grid 4P streaming MCTS value diagnostic | Resubmitted smaller inference-only MCTS on the original L1 checkpoint after adding per-example JSONL streaming and score caching. Tasks: learned `goal_energy` and oracle `latent_goal` at depths 4/8, 32 boards, 128 simulations, expansion cap 32. | Completed as `3715249_[0-3]`, exit `0:0`. Learned `goal_energy` d4/d8 solved `0/32`, terminal `0`, mean remaining Hamming `47.78`/`48.72`; oracle `latent_goal` d4/d8 solved `0/32`, terminal `0`, mean remaining Hamming `9.88`/`10.03`. Root debug top-1 goal-value writes: learned `65/438`, `70/452`; oracle `376/449`, `360/440`. |
| Grid 4Q recursive hierarchy planner on Grid 4M | New exact recursive planner diagnostic: level-2 optimizer proposes the first latent subgoal, level 1 recursively plans to that subgoal, and level 0 primitive CEM plans to the level-1 subgoal. Crosses top optimizer `{cem,gd,gd_reachability}` with Grid 4M methods. | Still pending as `3715252_[0-11]` but will never start: Slurm reports `DependencyNeverSatisfied` because Grid 4M task `_0` timed out. No recursive Grid 4Q artifacts exist. Resubmit without the failed array dependency if this read is still desired. |
| Grid 4R recursive hierarchy planner on Grid 4N | Same recursive planner diagnostic for the true macro-action-advantage checkpoint, crossing `{cem,gd,gd_reachability}` with top score `macro_action_advantage`. | Completed as `3715251_[0-2]`, exit `0:0`. Recursive `cem`, `gd`, and `gd_reachability` all solved `0/16`, terminal `0/16`; mean remaining Hamming was `51.81`, `52.94`, and `51.31`. The learned macro-action top score is not directionally useful in this recursive read. |
| Local A100 4M/4N qualitative value/MCTS probe | Reproducible local analysis via `scripts/analysis/sudoku_hier_value_probe.py`; artifacts in `/home/vault/c107fa/c107fa12/sequence-editing/analysis/*20260611.json`. | Completed on 2026-06-11. Terminal scoring is coarse-useful but not fine-safe; local successor ranking remains weak; high-level latent subgoals are not reachable by low-level CEM; terminal-depth MCTS reaches only depth `2-4` at 256/1024 simulations and sees no terminal leaves. |
| Local A100 terminal-projection/on-manifold follow-up | Reproducible local analysis via `scripts/analysis/sudoku_terminal_projection_probe.py`; artifacts in `/home/vault/c107fa/c107fa12/sequence-editing/analysis/sudoku_terminal_projection_*_20260611.json`. Tests oracle terminal completions after each first action, high-level nearest-terminal latents, and real-chunk macro-action codebooks. | Completed on 2026-06-11. Even with oracle terminal completion, true-solution leaves are never top-1 globally for Grid 4M `state_value` or `terminal_energy`; mean best-solution ranks are `156.38` and `64.06`. Top hierarchy nearest-terminal analysis also has true terminal top-1 `0.0`, and real-chunk codebook restriction does not select the oracle chunk. |
| Grid 4S HWM-style macro-action bottleneck/codebook | Completed cleanly as `3717328_[0-9]`; all ten tasks exited `0:0` with empty stderrs. Training-only L3 span-4 grid with macro dims `4/8/16/32`, VQ variants `(4,64)` and `(8,128)`, plus `state_value` and `macro_action_advantage` scorer variants. | Wrapper: `scripts/slurm/run_grid4s_macro_bottleneck_l3.slurm`. Training losses are finite, but the associated planner matrix failed the solve gate. |
| Grid 4T macro-bottleneck planner matrix | Completed cleanly as `3717329_[0-57]`; all 58 eval tasks exited `0:0` with empty stderrs. Eval-only array for Grid 4S checkpoints: flat symbolic MCTS to depth `32` with 256 simulations, recursive L1 hierarchy CEM, and recursive full L2/3-level hierarchy CEM. | Every planner read solved `0`. Best oracle MCTS mean remaining Hamming was `8.56`; learned-score and recursive CEM reads stayed around `49-52`. Bottleneck/codebook macro actions did not create usable hierarchy here. |
| Grid 4U global single-latent MLP JEPA | Completed cleanly as `3717900_[0-2]`: L1 elapsed `00:13:56`, L2 `00:24:43`, L3 `00:25:06`, all on `a2141`. Training-only global-board MLP variants with `hierarchy_levels={1,2,3}`, one latent token for the whole Sudoku board, no attention, no CLS token, no cell marker, compact `action_embedding_dim=32`, and goal-energy regression. | Wrappers/configs: `scripts/slurm/run_grid4u_global_mlp_latent.slurm` and `configs/puzzle/grid4u_sudoku_global_mlp_l{1,2,3}.yaml`. Planner results in 4V/4W are negative. |
| Grid 4V global single-latent planner matrix | Completed cleanly as `3717901_[0-15]`; all stderrs are empty. h16 MPC-CEM solves `0/64` for L1/L2/L3 under both learned `goal_energy` and oracle `latent_goal`; reset/beam oracle `latent_goal` also solves `0/128`. | Current read: the global bottleneck did not fix learned scoring and damaged oracle-goal planning geometry. Best re-encoded oracle reset Hamming is `19.05`, still far from tokenized oracle `128/128`. |
| Grid 4W global single-latent long-horizon MPC-CEM | Completed cleanly as `3718124_[0-11]`; all stderrs are empty. h32/h64 MPC-CEM solves `0/64` under learned and oracle scores for all L1/L2/L3 checkpoints. | Longer horizon did not rescue the global MLP latent. |
| Grid 4X mixed rollout / mixed hierarchy global MLP | Completed cleanly as `3718216_[0-3]`: K2 elapsed `00:13:12`, K4 `00:15:02`, L2 mixed hierarchy `00:15:56`, L3 mixed hierarchy `00:20:57`; stderrs are empty. Training losses are finite. Rollout/hierarchy trajectory batches use `oracle_probability=0.5`, so half are correct trajectories and half are coherent wrong/random trajectories. | Wrappers/configs: `scripts/slurm/run_grid4x_global_mlp_mixed_rollout.slurm` and `configs/puzzle/grid4x_sudoku_global_mlp_*`. Training completed; planner eval is Grid 4Y. |
| Grid 4Y mixed rollout planner matrix | Completed cleanly as `3718217_[0-19]`; all stderrs are empty. h16/h32 MPC-CEM solves `0/64` under learned and oracle scores for all mixed checkpoints; recursive hierarchy CEM solves `0/24` for L2/L3. | Wrapper: `scripts/slurm/run_grid4y_global_mlp_mixed_rollout_eval.slurm`. Mixed wrong rollouts did not rescue global MLP planning. |
| Grid 4P/4Q/4R one-shot oversight | User-requested non-recurring checks at 6h, 10h, 12h, and 14h after submission. They should inspect jobs/logs/partial JSONL, fix clear bugs, analyze results, iterate only if needed, update docs, and commit/push. | All four original watches began together at 2026-06-10 11:42:38 CEST. Stale duplicate active watch jobs `3715250`, `3715254`, and `3715255` were cancelled at 11:44:50 with logs preserved; stale running watch `3715253` was cancelled at 11:56:08 after it cancelled the first new scheduled attempt `3715429`-`3715433` before start. |
| Grid 4P/4Q/4R cancelled scheduled attempt | First attempt at the exact Europe/Berlin checks requested for 2026-06-10 18:00/20:00 and 2026-06-11 00:00/04:00/08:00. | Submitted as `3715429`, `3715430`, `3715431`, `3715433`, and `3715432`, but stale watch `3715253` cancelled them before start at 11:53:40 CEST. Superseded by `3715446`-`3715450`. |
| Grid 4P/4Q/4R replacement scheduled oversight | Replacement for the exact Europe/Berlin checks after stale watch `3715253` cancelled the first attempt. | `3715446`-`3715450` all completed cleanly. The final 08:00 check `3715450` ran 2026-06-11 08:00:26-08:24:05 CEST on `a1621` and confirmed proxy inheritance; no successor oversight job was submitted. |
| Grid 4O one-shot oversight | User-requested non-recurring Codex checks for Grid 4O/4M/4N health, logs, results, bug fixes, analysis, and handoff updates. | `3714106`, `3714107`, and `3714108` completed cleanly. They did not submit successors. |
| Planner-state reset/re-encoding branch | Keep symbolic candidate boards as planner state of record and re-encode latents every 4 actions for scoring. | Keep as oracle-goal control/baseline for Grid 4A; do before Maze, broad controls, or model-size sweeps if Grid 4A fails the non-oracle energy gate. |

Grid 3A Sudoku local-edit ablation:

| Run | Prediction | Loss | Status |
| --- | --- | --- | --- |
| `sudoku_jepa_5m_local_direct_uniform` | direct next latent | uniform | Completed as `3674778_0`, step `5000`, online solve `1.0 / 1.0 / 1.0` |
| `sudoku_jepa_5m_local_direct_weighted` | direct next latent | changed cell high, row/col/block medium | Completed as `3674778_1`, step `5000`, online solve `1.0 / 1.0 / 1.0` |
| `sudoku_jepa_5m_local_residual_weighted` | `z_next = z_current + delta` | same weighted loss | Completed as `3674778_2`, step `5000`, online solve `0.0 / 0.0 / 0.0` |
| `sudoku_jepa_5m_local_direct_changed_only` | direct next latent | changed-cell token only | Completed as `3674778_3`, step `5000`, online solve `0.0 / 0.0 / 0.0` |

Dependent diagnostics `3674779_[0-3]` failed on CLI argument formatting before
model load. The wrapper was fixed and diagnostics were resubmitted as
`3676904_[0-3]`; they completed successfully.

## Gate

Grid 3A diagnostic decision:

1. Direct local injection passes the action-grounding gate: direct uniform and
   direct weighted both have diagnostic `goal_rank` mean/top1 `1.0`.
2. Direct weighted is the preferred follow-up seed: it has lower short drift
   than uniform and better terminal-planning proximity, despite slightly worse
   single-oracle rank.
3. Residual is rejected for the next branch because rollout drift explodes
   (`drift@20 103`, terminal `1940`).
4. Changed-cell-only loss is rejected except as a negative control because
   `goal_rank` and planning are poor.
5. Grid 3B lead diagnosis: terminal failure is mostly latent rollout drift
   under the oracle-goal diagnostic. Re-encoded symbolic-state planning solves
   all 64 boards, while latent rollout planning solves none; terminal-only
   scoring does not materially improve latent planning.
6. Grid 3B rollout `N=2` preserves sampled `goal_rank=1.0` and improves
   proximity, but it does not satisfy the exact latent solve gate: latent
   terminal-energy solve is only `4/64` and terminal weighted drift remains
   about `2.16`.
7. Grid 3C passed the mechanism gate: periodic re-encoding can recover the
   `64/64` re-encoded result on the paired oracle-goal diagnostic.
8. Grid 3D confirmed the mechanism on 128 paired boards: reset every 4 solved
   `128/128` under both step- and terminal-energy selection; reset every 8
   solved `91/128` under step-energy and `128/128` under terminal-energy
   selection.
9. User approved cancelling the pre-correction array; `3688587_[0-2]` was
   cancelled. Intermediate corrected training `3688921_[0-2]` was also cancelled
   after the user asked for the exact report-style planner. The implementation
   now has explicit higher-level action encoders, configurable `hierarchy_span`,
   continuous high-level latent-action CEM, and low-level primitive CEM to reach
   the first predicted latent subgoal. Replacement training `3688986_[0-2]`
   and diagnostics finished cleanly, but learned-energy CEM solved
   `0/64` for all levels and exact subgoal CEM solved `0/32` for L2/L3. Current
   gate: Grid 4B shows the learned goal-energy head does not work even under
   the beam/reset regime. Prioritize energy-head ranking/calibration or a
   verifier/goal objective before changing CEM, starting Maze, broad controls,
   or model-size sweeps.
10. Grid 4C `3695040` passed the sanity check: the L1 Grid 4A checkpoint still
    solves `128/128` with oracle solved-board latent MSE under reset every 4 and
    re-encoded planning. This isolates the learned goal-energy scorer as the
    blocker. Calibration is close in aggregate but imperfect locally: reset
    every 4 terminal trajectories have mean absolute predicted-vs-true latent
    distance error about `0.010`, predicted energy monotonicity about `0.923`,
    and true latent distance monotonicity `1.0`.
11. Grid 4D `3696616_[0-5]` tests whether the energy head needs local
    negatives rather than pure scalar regression. The variants are `nce`,
    `infonce`, `margin`, `nce_mono`, `infonce_mono`, and `margin_mono`.
    Regression remains on the normal mixed batch; contrastive positives and
    monotonicity use an oracle-only auxiliary batch, with 8 local successor
    negatives sampled from the same Sudoku overwrite/conflict action space used
    by learned-energy beam planning. Gate: learned-energy reset/beam should
    solve nonzero boards and ideally approach the oracle-goal reset control.
    Superseded submission `3696588_[0-5]` failed before training because Hydra
    requires `+training.*` syntax for keys absent from the base YAML.
    Superseded `3696609_[0-5]` fixed that but failed before checkpointing
    because `512` auxiliary examples times `16` negatives was too memory-heavy;
    the live array uses auxiliary batch `64`, `8` negatives, and staggered
    starts. At 09:24 CEST on 2026-06-04, all six variants had final
    checkpoints and learned-energy reset/beam diagnostics. Learned-energy
    reset/beam solved `0/128` for every variant. Mean remaining Hamming under
    paired reset was `47.74` (InfoNCE), `47.27` (InfoNCE+mono), `51.20`
    (margin), `48.22` (margin+mono), `53.14` (NCE), and `52.64`
    (NCE+mono). The jobs are still running because the oracle-goal
    reset/calibration control has not produced outputs yet.
12. Grid 4E `3698281_[0-6]` is an analysis-only diagnostic, not a training
    run. It exhaustively enumerates every legal Sudoku overwrite/conflict
    planning action at each sampled oracle step, scores the resulting successor
    state with learned goal energy, and reports the gold action rank plus
    margins against same-square wrong numbers, other-square goal-correct
    actions, and other-square wrong actions. This differs from Grid 4D training,
    which sampled 8 negatives per auxiliary example for memory reasons.
13. Literature read on non-RL value analogs: MuZero/Dreamer/TD-MPC value heads
    are not the clean recipe because they rely on reward, TD, or search targets.
    JEPA/V-JEPA gives latent prediction and image-goal planning, but not a
    standalone supervised value. The closest fit is contrastive
    goal-conditioned reachability/value learning. For the next scorer design,
    use multi-positive future/reachable successors: in Sudoku, all currently
    wrong mutable cells filled with their goal value should be positives, and
    wrong fills should be negatives.
14. Grid 4F `3698394_[1-2]` implements the first follow-up to that literature
    read after cancelling the value-guided JEPA task. One-step smokes passed
    before submission. The eval gate is learned-energy reset/beam on 128 boards
    plus the oracle-goal reset control.
15. Grid 4G `3698893` tests the user's batch-structure hypothesis directly:
    keep auxiliary batch size `64`, but sample it as `16` puzzles x `4`
    states/puzzle, with `8` positives and `32` negatives per state. Local
    one-step smoke hung in startup/import and was killed; compile and Slurm
    syntax checks passed, and Slurm stderr was empty at startup.
16. Grid 4H `3698988` was cancelled because direct terminal correctness was too
    sparse: all reachable nonterminal boards had label `0`.
17. Grid 4I `3699523` is the corrected dense value experiment. It keeps JEPA
    dynamics training, disables scalar latent-energy regression, and trains the
    existing scalar head with soft BCE targets `0.99^N`, where `N` is the
    remaining wrong-cell count to the solution.
18. Grid 4K `3702254_[0-1]` is the next scorer experiment. It uses ListNet over
    sampled local successor lists so the head is trained on relative action
    ordering directly. The two label variants test task-native discounted
    remaining-wrong-cell relevance versus oracle latent goal-distance relevance.
