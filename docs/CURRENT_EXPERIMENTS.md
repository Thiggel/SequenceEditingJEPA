# Current Experiments

Source of truth: `../sequence-editing-report/CURRENT_EXPERIMENTS.md`.

# Current Experiments

Last updated: 2026-07-06 09:25 CEST

## Active: Verifier-Free Energy Sweep

Purpose:
- Replace oracle/predicted goal latent scoring with learned verifier-like
  signals: compatibility energy `W` for wrong commitments and remaining-edit
  head `R` for progress.
- Keep the base fixed to full-grid EMA+VICReg, editable cells,
  counterfactual branches, K8 smooth/count dense rollout, and affected-context
  dynamics weighting.

Prepared variants:

| Variant | Purpose |
|---|---|
| `E0_base_oracle_sanity` | Oracle raw-L2 sanity baseline with no verifier heads. |
| `E1_compat_state` | W only on encoded states plus corruptions. |
| `E2_remaining_state` | R only on encoded states plus corruptions. |
| `E3_wr_state` | W+R on encoded states. |
| `E4_wr_predicted` | Add W/R supervision on predicted successor latents. |
| `E5_wr_pairwise_rank` | Add pairwise latent-successor ranking. |
| `E6_wr_listwise_rank` | Add listwise latent-successor ranking. |
| `E7_wr_listwise_policy` | Add verifier-targeted policy prior. |
| `E8_wr_no_counterfactual` | Remove counterfactual dynamics branches from the full scorer. |
| `E9_wr_no_corruption` | Remove synthetic corruption negatives from the full scorer. |
| `F0_full_score` | Full W+R score without policy prior. |
| `F1_full_policy` | Full W+R score with verifier-targeted policy prior. |

Prepared scripts:
- `scripts/slurm/run_grid_goal_verifier_energy_train.slurm`
- `scripts/slurm/run_grid_goal_verifier_energy_eval.slurm`
- `scripts/experiments/submit_grid_goal_verifier_energy.sh`

State:
- Submitted on 2026-07-06 at about 09:24 CEST.
- All 12 train jobs started immediately on `rtxpro6k`.
- All 12 eval jobs are dependency-held individually behind their matching train
  job.

Jobs:

| Variant | Train | Eval | Purpose |
|---|---:|---:|---|
| `E0_base_oracle_sanity` | `3815607` | `3815608` | Oracle raw-L2 sanity baseline with no verifier heads. |
| `E1_compat_state` | `3815609` | `3815610` | W only on encoded states plus corruptions. |
| `E2_remaining_state` | `3815611` | `3815612` | R only on encoded states plus corruptions. |
| `E3_wr_state` | `3815613` | `3815614` | W+R on encoded states. |
| `E4_wr_predicted` | `3815615` | `3815616` | Add W/R supervision on predicted successor latents. |
| `E5_wr_pairwise_rank` | `3815617` | `3815618` | Add pairwise latent-successor ranking. |
| `E6_wr_listwise_rank` | `3815619` | `3815620` | Add listwise latent-successor ranking. |
| `E7_wr_listwise_policy` | `3815621` | `3815622` | Add verifier-targeted policy prior. |
| `E8_wr_no_counterfactual` | `3815623` | `3815624` | Remove counterfactual dynamics branches from the full scorer. |
| `E9_wr_no_corruption` | `3815625` | `3815626` | Remove synthetic corruption negatives from the full scorer. |
| `F0_full_score` | `3815627` | `3815628` | Full W+R score without policy prior. |
| `F1_full_policy` | `3815629` | `3815630` | Full W+R score with verifier-targeted policy prior. |

Audit blockers fixed on 2026-07-06:
- `verifier_energy` MPC no longer encodes the oracle goal latent during setup.
- Sequence rank-state sampling with `allow_overwrite=True` selects filled-wrong
  recovery states instead of blank-only frames.
- Listwise verifier-targeted policy prior trains overwrite recovery actions on
  filled-wrong boards with no blanks.
- Single-CLS compatibility supervision uses binary BCE labels while retaining
  count regression for the wrong-count target.

## Active: Wide Single-CLS Oracle Probe

Purpose:
- Test whether the one-vector board latent failed because `d_model=256` was too narrow, or because the one-vector geometry/predictor is structurally wrong for Sudoku.
- Keep the readout oracle-only for now: no predicted goal, no waypoint, no value head.

Runs:

| Variant | Train | Eval | Stabilization |
|---|---:|---:|---|
| `W0_ema_vicreg_d1024` | `3815481` | `3815482` | EMA + VICReg, no LDAD |
| `W1_ema_ldad_set_d1024` | `3815483` | `3815484` | EMA + LDAD, no VICReg |
| `W2_ldad_vicreg_set_d1024` | `3815485` | `3815486` | LDAD + VICReg, no EMA |
| `W3_ldad_only_set_d1024` | `3815487` | `3815488` | LDAD only, no EMA/VICReg |

Latest state at 2026-07-06 09:25 CEST:
- train jobs `3815481`, `3815483`, `3815485`, and `3815487` are running on
  `rtxpro6k` node `a2841`
- eval jobs `3815482`, `3815484`, `3815486`, and `3815488` are
  dependency-held
- latest logged progress: `W0` step `1000`, `W1` step `1000`, `W2` step
  `500`, `W3` step `500` out of `5000`

Common setup:
- `latent_representation=single`
- `d_model=1024`, `num_heads=16`, `distance_dim=256`
- 5k steps, effective batch 8 via micro-batch 2 and grad accumulation 4
- counterfactual editable data, K8 smooth/count dense rollout, affected-context dynamics weighting
- eval: oracle raw L2 only, `mpc_beam`, latent rollout and symbolic re-encode, depths `{4,16}`, 8 boards
- output root: `$PUZZLE_JEPA_WORK_ROOT/runs/grid_goal_single_wide`

Storage housekeeping:
- `$HPCVAULT/sequence-editing` was reduced from about `759G` to `5.9G`
- removed redundant `checkpoint-*.pt`, final old `checkpoint.pt`, HF-style model/optimizer artifacts, vault cache, and failed zero-byte single-wide files
- preserved lightweight configs, metrics, diagnostics, and planner/result records
- cancelled stale sequence-editing dependency-never-satisfied evals and old weekend oversight jobs; current wide-single train/eval jobs were not touched

Gate:
- If `W0` solves but LDAD-only variants fail, width helps single-CLS only with EMA/VICReg.
- If `W3` solves under latent rollout, prior Delta/single failure was probably capacity-limited.
- If all four fail, the single-vector board latent is likely structurally poor for this planner.

## Completed Context: Counterfactual Editable Weekend Wave

Research questions:
- Does counterfactual branching improve action dependence and latent-rollout top-action accuracy?
- Does making non-given cells editable reduce the irreversibility/asymmetric-distance failure mode?
- Which action conditioning works best once counterfactual branches and editable repairs are present?
- Does receding-horizon waypoint prediction beat one-shot terminal predicted-goal planning?
- When predicted-waypoint solve rate is zero, are intermediate waypoints nevertheless locally correct, progressive, or trackable?
- Do asymmetric source/goal projections or value-guided quasi-distance improve non-oracle planning?
- Can Delta-JEPA work once data coverage and action conditioning are fixed?
- Which isolated winners combine constructively in an integrated recipe?

Current summary:
- run suffix: `_mb4ga2`
- oversight cadence: `6` hours
- repair evals enabled: `True`
- completed checkpoints: 30
- total planner rows: 326
- final interpretation: oracle full-grid geometry works; predicted goals/waypoints and faithful macro hierarchy do not.

Insights:
- Best current row: mpc_beam latent_rollout oracle_goal_raw_euclidean_distance solve=8/8 h=0.0
- Delta branch has eval rows; compare every grid variant against its single-CLS paired variant before promoting.
- Waypoint rows are present; prioritize predicted-waypoint versus oracle-waypoint gap before terminal predicted-goal variants.
- Current waypoint rows may be flat-only; do not treat waypoint_beam solves as evidence that hierarchical MPC works.
- If predicted-waypoint solve rate is still zero, inspect waypoint quality directly: latent alignment to oracle future waypoints, Hamming progress after one tracked chunk, and trackability distance.
- Final weekend conclusion: counterfactual/editable training preserved strong oracle latent-rollout MPC but did not fix non-oracle planning. Macro CEM/MPPI hierarchy rows also failed even with oracle waypoints.

Required oversight diagnostics:
- predicted waypoint latent alignment: compare q_hat_H(s_t) with oracle future waypoint E(s_{min(t+H,T)}) by raw L2/cosine and report the oracle-vs-predicted gap
- predicted waypoint progress: after tracking q_hat_H for one MPC chunk, report Hamming/edit-distance improvement toward the solved board even if no full solve occurs
- predicted waypoint trackability: report D(E(s_after_mpc), q_hat_H) and compare it to D(E(s_t), q_hat_H)
- multi-horizon consistency: for multi-waypoint heads, check whether predicted H4/H8/H16 waypoints are increasingly closer to their matching oracle future waypoints than to mismatched horizons
- terminal-locality split: report waypoint quality separately for early, middle, and near-terminal states
- hierarchical waypoint rows must include macro-action CEM or MPPI tracking; flat waypoint_beam rows are only primitive-tracker baselines

Variant table:

| Variant | Checkpoint | Rows | Best result |
|---|---:|---:|---|
| `S0_anchor_olddata` | `True` | `8` | mpc_beam / latent_rollout / oracle_goal_raw_euclidean_distance = 8/8, h 0.0 |
| `S1_counterfactual_fill` | `True` | `8` | mpc_beam / latent_rollout / oracle_goal_raw_euclidean_distance = 8/8, h 0.0 |
| `S2_counterfactual_edit` | `True` | `8` | mpc_beam / latent_rollout / oracle_goal_raw_euclidean_distance = 8/8, h 0.0 |
| `S3_counterfactual_edit_adaln` | `True` | `8` | mpc_beam / latent_rollout / oracle_goal_raw_euclidean_distance = 8/8, h 0.0 |
| `S4_counterfactual_edit_oldlocal` | `True` | `8` | mpc_beam / latent_rollout / oracle_goal_raw_euclidean_distance = 8/8, h 0.0 |
| `E0_base_cf_edit` | `True` | `8` | mpc_beam / latent_rollout / oracle_goal_raw_euclidean_distance = 8/8, h 0.0 |
| `E1_hierarchy_l4_l8_l16` | `True` | `9` | mpc_beam / latent_rollout / oracle_goal_raw_euclidean_distance = 8/8, h 0.0 |
| `E2_waypoint_h8` | `True` | `8` | waypoint_beam / latent_rollout / oracle_waypoint_raw_euclidean_distance = 8/8, h 0.0 |
| `E3_waypoint_h16` | `True` | `8` | waypoint_beam / latent_rollout / oracle_waypoint_raw_euclidean_distance = 8/8, h 0.0 |
| `E4_waypoint_h4_h8_h16` | `True` | `8` | waypoint_beam / latent_rollout / oracle_waypoint_raw_euclidean_distance = 8/8, h 0.0 |
| `E5_waypoint_h16_hierarchy` | `True` | `25` | waypoint_beam / latent_rollout / oracle_waypoint_raw_euclidean_distance = 8/8, h 0.0 |
| `D0_online_h1_grid` | `True` | `4` | mpc_beam / symbolic_reencode / oracle_goal_raw_euclidean_distance = 8/8, h 0.0 |
| `D0_online_h1_single` | `True` | `4` | mpc_beam / symbolic_reencode / oracle_goal_raw_euclidean_distance = 0/8, h 45.625 |
| `D1_online_ordered_h12345_grid` | `True` | `4` | mpc_beam / symbolic_reencode / oracle_goal_raw_euclidean_distance = 8/8, h 0.0 |
| `D1_online_ordered_h12345_single` | `True` | `4` | mpc_beam / symbolic_reencode / oracle_goal_raw_euclidean_distance = 0/8, h 44.625 |
| `D2_online_set_h12345_grid` | `True` | `4` | mpc_beam / symbolic_reencode / oracle_goal_raw_euclidean_distance = 7/8, h 0.125 |
| `D2_online_set_h12345_single` | `True` | `7` | mpc_beam / symbolic_reencode / oracle_goal_raw_euclidean_distance = 0/8, h 44.25 |
| `D3_hybrid_ema_h1_grid` | `True` | `4` | mpc_beam / latent_rollout / oracle_goal_raw_euclidean_distance = 8/8, h 0.0 |
| `D3_hybrid_ema_h1_single` | `True` | `8` | mpc_beam / symbolic_reencode / oracle_goal_raw_euclidean_distance = 0/8, h 43.0 |
| `D4_hybrid_ema_set_h12345_grid` | `True` | `4` | mpc_beam / latent_rollout / oracle_goal_raw_euclidean_distance = 8/8, h 0.0 |
| `D4_hybrid_ema_set_h12345_single` | `True` | `4` | mpc_beam / symbolic_reencode / oracle_goal_raw_euclidean_distance = 0/8, h 41.875 |
| `D5_ema_vicreg_no_ldad_grid` | `True` | `4` | mpc_beam / latent_rollout / oracle_goal_raw_euclidean_distance = 8/8, h 0.0 |
| `D5_ema_vicreg_no_ldad_single` | `True` | `4` | mpc_beam / symbolic_reencode / oracle_goal_raw_euclidean_distance = 0/8, h 45.25 |
| `V1_asym_hindsight` | `True` | `8` | mpc_beam / latent_rollout / oracle_goal_projected_euclidean_distance = 8/8, h 0.0 |
| `V2_iql_quasi` | `True` | `8` | mpc_beam / latent_rollout / predicted_goal_projected_euclidean_distance = 0/8, h 55.375 |
| `V3_waypoint_asym_hindsight` | `True` | `31` | waypoint_beam / latent_rollout / oracle_waypoint_raw_euclidean_distance = 8/8, h 0.0 |
| `I0_integrated_waypoint_asym` | `True` | `31` | waypoint_beam / latent_rollout / oracle_waypoint_raw_euclidean_distance = 8/8, h 0.0 |
| `I1_integrated_waypoint_iql` | `True` | `30` | waypoint_hierarchical_beam / symbolic_reencode / oracle_waypoint_raw_euclidean_distance = 8/8, h 0.0 |
| `I2_integrated_best_delta_if_gate_passes_grid` | `True` | `27` | waypoint_beam / symbolic_reencode / oracle_waypoint_raw_euclidean_distance = 5/8, h 0.625 |
| `I2_integrated_best_delta_if_gate_passes_single` | `True` | `30` | waypoint_beam / symbolic_reencode / oracle_waypoint_raw_euclidean_distance = 0/8, h 24.125 |
