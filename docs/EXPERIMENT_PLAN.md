# Experiment Plan

Last updated: 2026-05-31 13:30 CEST

The active backlog now lives in `../sequence-editing-report/BACKLOG.md`.

## Active Experiments

Grid 3B Sudoku follow-up:

| Run | Purpose | Status |
| --- | --- | --- |
| `sudoku_jepa_5m_local_direct_weighted` large diagnostics | Increase eval sample size and compare latent rollout planning with re-encoded symbolic-state planning; write terminal board records. | Completed as `3680019`; re-encoded planning solved `64/64`, latent rollout solved `0/64`. |
| `sudoku_jepa_5m_local_direct_weighted_rollout_n2` | Train direct local weighted JEPA with rollout loss `N=2`. | Completed as `3680020`; final step `5000`, eval loss `0.000138`, online H1/H2/H4 solve `1.0 / 1.0 / 1.0`. |
| Grid 3B rollout `N=2` diagnostics | Same larger diagnostics after rollout training. | Completed as `3680021`; latent terminal-energy solve `4/64`, re-encoded planning `64/64`. |
| Grid 3C reset/re-encoding diagnostic | Test periodic candidate-state re-encoding or latent reset cadence before broad scaling. | Completed as `3682924`; reset every 2/4 solved `64/64` paired boards under step and terminal energy, while no-reset terminal energy solved `2/64`. |
| Grid 3D reset-large confirmation | Confirm the reset/re-encoding branch on a larger paired sample before changing planner defaults or scaling. | Running as `3683903`; no `diagnostics_reset_cadence_large/` directory yet as of 13:30 CEST. |
| Enhanced recurring oversight | Every run audits jobs, examples, assumptions, figures/tables, backlog gates, and next submissions. | `3683863` completed; `3684237` is running; successor `3684889` is pending for `2026-05-31 17:27:28 CEST`. |

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
7. Current gate: do not start Maze, broad size sweeps, or broad controls yet.
   Grid 3C passed the mechanism gate: periodic re-encoding can recover the
   `64/64` re-encoded result on the paired oracle-goal diagnostic.
8. Next gate: analyze Grid 3D job `3683903` when it completes. If reset every 4
   remains exact on the larger paired sample, pursue a planner-state
   reset/re-encoding branch before Maze, broad controls, or model-size sweeps.
