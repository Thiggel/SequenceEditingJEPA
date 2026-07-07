# Current Experiments

Source of truth: `../sequence-editing-report/CURRENT_EXPERIMENTS.md`.

Last updated: 2026-07-07 12:05 CEST

## Structured JEPA Wave

Status: implemented and script-prepared, not submitted.

Purpose: test the next architectural hypothesis one component at a time after
single-CLS, predicted-goal, waypoint, Delta-JEPA, and verifier-free W/R waves
failed to produce non-oracle solves. This wave keeps the reliable full-grid
Sudoku base and adds structured latent slots, Delta-JEPA delta-source variants,
SD-JEPA-style progress subspace supervision, preference/action ranking, and a
goal+waypoint planner score.

Prepared scripts:

- `scripts/slurm/run_grid_goal_structured_wave_train.slurm`
- `scripts/slurm/run_grid_goal_structured_wave_eval.slurm`
- `scripts/experiments/submit_grid_goal_structured_wave.sh`

Prepared variants:

| Block | Variants | What it tests |
|---|---|---|
| Structured slots | `S0`-`S4` | 81 cells versus unit/global/progress/full slot layouts. |
| Delta-JEPA | `DJ0`-`DJ5` plus paired `_single` rows | Action conditioning crossed with all-token versus changed-cell+unit LDAD sources; every full-grid Delta row has a learned-CLS single-latent counterpart. |
| SD-JEPA | `SD0`-`SD3` | Separate progress projection and action-effect subspace. |
| Preference ranking | `PR0`-`PR4` | State progress rank, legal/listwise action rank, and predictor-successor ranking for PR2. |
| Goal/waypoint | `GW0`-`GW4` | Terminal goal, waypoint, waypoint+goal score, goal-conditioned waypoint, and multi-waypoint sketch. |

No Slurm job IDs exist yet. The submitter now creates 31 training jobs and 31
dependency-held individual eval jobs. Eval runs diagnostics first, including
LDAD action-delta probes, delta-locality probes, and SD-progress ordering
probes; goal/waypoint rows include the combined
`predicted_waypoint_goal_raw_euclidean_distance` score.

## Wide Single-CLS Oracle Probe

Purpose: test whether one-vector board latents fail only because `d_model=256`
was too narrow. These runs use `d_model=1024`, learned CLS board state, causal
history predictor, oracle raw-L2 planning only, and no predicted goal/value head.

Result: all four train jobs completed. All four eval jobs hit the 8h walltime
after writing three shallow rows each. No variant solved any board.

| Variant | Train | Eval | Rows | Best result |
|---|---:|---:|---:|---|
| `W0_ema_vicreg_d1024` | `3817223` completed | `3817224` timeout/partial | 3 | `0/8`, h `54.125`, symbolic re-encode depth 4 |
| `W1_ema_ldad_set_d1024` | `3817225` completed | `3817226` timeout/partial | 3 | `0/8`, h `31.25`, symbolic re-encode depth 4 |
| `W2_ldad_vicreg_set_d1024` | `3817227` completed | `3817228` timeout/partial | 3 | `0/8`, h `34.125`, symbolic re-encode depth 4 |
| `W3_ldad_only_set_d1024` | `3817229` completed | `3817230` timeout/partial | 3 | `0/8`, h `32.875`, symbolic re-encode depth 4 |

Interpretation: widening the single latent helps symbolic re-encode hamming for
LDAD variants versus EMA+VICReg, but latent rollout remains near random
(`h ~= 54-55`) and oracle planning still does not solve. The one-vector path is
not competitive with the full-grid oracle recipe yet.

## Verifier-Free Energy Repair Wave

Purpose: repair the learned verifier-free W/R planner after the first sweep
showed oracle raw-L2 solved but learned compatibility/progress scoring did not.
This wave tested sampling hardness, W/R/rank loss scale, energy projections,
predicted-latent calibration, and local same-parent action discrimination.

Result: 16 variants produced planner/diagnostic results or partial results.
15 variants OOMed during training; their dependency-held eval placeholders were
canceled after Slurm marked them dependency-never-satisfied. No completed
learned verifier-free variant solved any board.

Best planner rows:

| Variant | Rows | Best result |
|---|---:|---|
| `B0_current_E4` | 18 | `0/8`, h `48.5`, remaining-count latent rollout depth 1 |
| `B1_current_F0` | 18 | `0/8`, h `51.75`, remaining-count symbolic re-encode depth 8 |
| `S1_balanced_partials` | 18 | `0/8`, h `49.0`, remaining-count latent rollout depth 8 |
| `S2_near_solution` | 18 | `0/8`, h `48.125`, remaining-count symbolic re-encode depth 8 |
| `S3_recovery_states` | 18 | `0/8`, h `49.25`, remaining-count latent rollout depth 1 |
| `S4_hard_action_sets` | 18 | `0/8`, h `50.625`, verifier-energy symbolic re-encode depth 8 |
| `W1_energy_x3` | 18 | `0/8`, h `48.25`, verifier-energy latent rollout depth 4 |
| `W2_energy_x10` | 13 | `0/8`, h `48.875`, verifier-energy symbolic re-encode depth 4 |
| `W3_energy_x30` | 18 | `0/8`, h `48.75`, remaining-count symbolic re-encode depth 4 |
| `W4_rank_x3` | 18 | `0/8`, h `51.0`, compatibility-energy latent rollout depth 8 |
| `W5_rank_x10` | 18 | `0/8`, h `51.5`, remaining-count symbolic re-encode depth 4 |
| `G1_energy_projection` | 18 | `0/8`, h `46.875`, remaining-count latent rollout depth 4 |
| `L5_exhaustive_small` | 18 | `0/8`, h `49.0`, compatibility-energy latent rollout depth 4 |
| `C1_sampling_weight` | 18 | `0/8`, h `50.125`, remaining-count symbolic re-encode depth 4 |
| `C2_sampling_rank` | 11 | `0/8`, h `50.375`, compatibility-energy latent rollout depth 8 |
| `C4_energy_proj_rank` | 18 | `0/8`, h `50.25`, remaining-count symbolic re-encode depth 1 |

Key diagnostics:

| Variant | Compat AUC | Remaining Spearman | Remaining MAE | Successor top1/top5 |
|---|---:|---:|---:|---:|
| `B0_current_E4` | `0.493` | `0.968` | `1.314` | `0.375/0.688` |
| `S1_balanced_partials` | `0.531` | `0.973` | `0.499` | `0.250/0.563` |
| `S2_near_solution` | `0.663` | `0.975` | `0.495` | `0.250/0.750` |
| `G1_energy_projection` | `0.524` | `0.973` | `1.765` | `0.563/0.875` |
| `L5_exhaustive_small` | `0.524` | `0.971` | `0.924` | `0.313/0.688` |
| `C2_sampling_rank` | `0.512` | `0.970` | `1.169` | `0.938/1.000` |

Interpretation: the remaining-edit scalar can be fitted on held-out encoded
states, but the score is not giving robust search behavior. Compatibility W is
mostly near chance; only near-solution sampling moves AUC materially above
chance. Several variants improve same-parent successor ranking, especially
`C2_sampling_rank`, but that still does not translate into beam solve rate.

Failed/OOM groups:
- Geometry fine-tune: `G2`, `G3`.
- Predicted-latent calibration: `P1`-`P4`.
- Local action rank except the small exhaustive row: `L1`-`L4`.
- Combined high-pressure rows: `C3`, `C5`-`C8`.

Output root:
`/home/atuin/c107fa/c107fa12/sequence-editing-repair-20260706/runs`.
