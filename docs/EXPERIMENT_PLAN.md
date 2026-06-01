# Experiment Plan

Last updated: 2026-06-01 18:35 CEST

The active backlog now lives in `../sequence-editing-report/BACKLOG.md`.

## Active Experiments

Grid 3B Sudoku follow-up:

| Run | Purpose | Status |
| --- | --- | --- |
| `sudoku_jepa_5m_local_direct_weighted` large diagnostics | Increase eval sample size and compare latent rollout planning with re-encoded symbolic-state planning; write terminal board records. | Completed as `3680019`; re-encoded planning solved `64/64`, latent rollout solved `0/64`. |
| `sudoku_jepa_5m_local_direct_weighted_rollout_n2` | Train direct local weighted JEPA with rollout loss `N=2`. | Completed as `3680020`; final step `5000`, eval loss `0.000138`, online H1/H2/H4 solve `1.0 / 1.0 / 1.0`. |
| Grid 3B rollout `N=2` diagnostics | Same larger diagnostics after rollout training. | Completed as `3680021`; latent terminal-energy solve `4/64`, re-encoded planning `64/64`. |
| Grid 3C reset/re-encoding diagnostic | Test periodic candidate-state re-encoding or latent reset cadence before broad scaling. | Completed as `3682924`; reset every 2/4 solved `64/64` paired boards under step and terminal energy, while no-reset terminal energy solved `2/64`. |
| Grid 3D reset-large confirmation | Confirm the reset/re-encoding branch on a larger paired sample before changing planner defaults or scaling. | Completed as `3683903`; reset every 4 solved `128/128`, reset every 8 solved `128/128` only under terminal-energy selection. |
| Grid 4A goal-energy / hierarchy / CEM | Train one-, two-, and three-level JEPA variants with a learned goal-energy head and evaluate with categorical CEM plus exact report-style hierarchical subgoal CEM. | Pre-correction `3688587_[0-2]` cancelled; intermediate `3688921_[0-2]` cancelled; replacement training `3688986_[0-2]` is running with L1 at step 3000 and L2/L3 at step 2000. Learned-energy CEM `3689396_[0-2]` is dependency-blocked on `3688986`; subgoal CEM `3689397_[0-1]` is dependency-blocked on `3689396`. |
| Planner-state reset/re-encoding branch | Keep symbolic candidate boards as planner state of record and re-encode latents every 4 actions for scoring. | Keep as oracle-goal control/baseline for Grid 4A; do before Maze, broad controls, or model-size sweeps if Grid 4A fails the non-oracle energy gate. |
| Enhanced recurring oversight | Every run audits jobs, examples, assumptions, figures/tables, backlog gates, and next submissions. | `3688542` completed; successor `3689344` was cancelled before start; replacement `3689685` is pending for `2026-06-01 22:35:52 CEST`. |

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
   the first predicted latent subgoal. Replacement training `3688986_[0-2]` is
   running with L1 at step 3000 and L2/L3 at step 2000. Current gate: wait for
   `3688986_[0-2]` to finish, then analyze queued learned-energy CEM
   `3689396_[0-2]` and queued subgoal CEM `3689397_[0-1]`. Do not start Maze,
   broad controls, or model-size sweeps.
