# Experiment Plan

Last updated: 2026-06-04 15:44 CEST

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
| Grid 4F value-method ablations | Test two literature-inspired scorer objectives on non-hierarchical L1: CVL multi-positive InfoNCE and MuZero-lite policy/value shaping. | Value-guided task `3698394_0` cancelled; active tasks `3698394_[1-2]` are running on two A100 nodes. |
| Grid 4G stratified CVL scorer | Same CVL objective as Grid 4F, but the auxiliary batch is structured as multiple states per puzzle: `16` puzzles x `4` states/puzzle. | Submitted as `3698893`; running on `a0532`. |
| Grid 4H terminal-correctness scorer | Replace scalar latent-energy regression with a direct balanced terminal-correctness target on the existing scalar head. | Submitted as `3698988`; running on `a0831`. |
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
16. Grid 4H `3698988` tests direct terminal correctness. It keeps JEPA dynamics
    training, disables the scalar latent-energy regression target, and trains
    the existing scalar head with balanced BCE on solved boards vs random
    mutable corruptions. The planner still uses the `goal_energy` score path,
    but now higher score means higher predicted terminal correctness.
