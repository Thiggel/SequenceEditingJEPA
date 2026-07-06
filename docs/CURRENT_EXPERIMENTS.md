# Current Experiments

Source of truth: `../sequence-editing-report/CURRENT_EXPERIMENTS.md`.

Last updated: 2026-07-06 17:50 CEST

## Active: Verifier-Free Energy Repair Wave

Purpose:
- Fix the first verifier-free energy result, where oracle raw-L2 still solved
  but learned W/R scoring gave `0/8`.
- Test isolated fixes for sampling hardness, W/R/rank loss scale, separate
  energy projection, predicted-rollout calibration horizons, and local
  same-parent action discrimination.
- Then test combined recipes that stack the most plausible fixes.

Implementation additions:
- `model.verifier_predicted_horizons` lets W/R supervise explicit rollout
  horizons such as `[1,2,3,4,5,6,7,8]`, not just one-step predicted latents.
- `model.verifier_energy_projection=mlp` adds a separate learned token
  projection before W/R heads, so energy geometry can be shaped without making
  the raw dynamics token space carry the scalar directly.
- New scripts:
  - `scripts/slurm/run_grid_goal_verifier_repair_train.slurm`
  - `scripts/slurm/run_grid_goal_verifier_repair_eval.slurm`
  - `scripts/experiments/submit_grid_goal_verifier_repair.sh`

Variants:

| Group | Variants |
|---|---|
| Baselines | `B0_current_E4`, `B1_current_F0` |
| Sampling/hardness | `S1_balanced_partials`, `S2_near_solution`, `S3_recovery_states`, `S4_hard_action_sets` |
| Loss weights | `W1_energy_x3`, `W2_energy_x10`, `W3_energy_x30`, `W4_rank_x3`, `W5_rank_x10` |
| Geometry pressure | `G1_energy_projection`, `G2_energy_finetune`, `G3_dyn_half` |
| Predicted-latent calibration | `P1_pred_h1`, `P2_pred_h4`, `P3_pred_h8`, `P4_score_consistency` |
| Local action discrimination | `L1_pair_same_cell`, `L2_pair_overwrite`, `L3_listwise_32`, `L4_listwise_128`, `L5_exhaustive_small` |
| Combined | `C1_sampling_weight`, `C2_sampling_rank`, `C3_pred_rank`, `C4_energy_proj_rank`, `C5_best_no_policy`, `C6_best_policy`, `C7_finetune_best`, `C8_full` |

Eval:
- 8 boards, beam width `16`, depths `{1,4,8}`.
- Transitions: latent rollout and symbolic re-encode.
- Scores: `verifier_energy`, `remaining_edit_count`, `compatibility_energy`.
- Diagnostics include compatibility AUC, remaining-edit Spearman/MAE, and
  same-parent successor top-k.

Gate:
- W AUC should move clearly above chance.
- Same-parent corrective action top-1/top-5 should improve.
- Symbolic-reencode W/R planning should improve before latent rollout.
- If symbolic W/R works but latent rollout fails, the next blocker is
  predicted-latent calibration.

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
- All 12 train jobs completed successfully on `rtxpro6k`.
- Evals `3815608`, `3815610`, and `3815612` completed successfully. Eval
  `3815624` timed out after writing partial planner rows. The remaining W/R
  evals are still running near the 6h walltime with partial rows on disk.
- Diagnostics show remaining-edit `R` learns a strong scalar signal in most
  variants (`remaining_spearman` about `0.93`-`0.97`, except the oracle-only
  sanity row), while compatibility `W` is still near-chance on the current tiny
  probe (`compatibility_auc` about `0.48`-`0.53`).
- Current planner readout is negative for learned verifier-free scoring:
  oracle raw-L2 sanity is `8/8`, h `0.0`, but every learned energy/progress row
  seen so far is `0/8`. Best non-oracle row so far is
  `E4_wr_predicted` with `remaining_edit_count`, latent rollout, depth `8`,
  h `48.0`. Most full-score/policy rows are around h `55`.

Current best rows:

| Variant | Best current row |
|---|---|
| `E0_base_oracle_sanity` | oracle raw L2, latent rollout, depth 1: `8/8`, h `0.0` |
| `E1_compat_state` | compatibility energy, symbolic re-encode, depth 8: `0/8`, h `54.5` |
| `E2_remaining_state` | remaining-edit count, symbolic re-encode, depth 8: `0/8`, h `49.0` |
| `E3_wr_state` | remaining-edit count, latent rollout, depth 16: `0/8`, h `48.75` |
| `E4_wr_predicted` | remaining-edit count, latent rollout, depth 8: `0/8`, h `48.0` |
| `E5_wr_pairwise_rank` | remaining-edit count, latent rollout, depth 8: `0/8`, h `48.375` |
| `E6_wr_listwise_rank` | compatibility energy, latent rollout, depth 8/16: `0/8`, h `55.375` |
| `E7_wr_listwise_policy` | verifier energy or remaining-edit, latent rollout, depth 1: `0/8`, h `55.25` |
| `E8_wr_no_counterfactual` | verifier energy, symbolic re-encode, depth 1: `0/8`, h `51.25` |
| `E9_wr_no_corruption` | verifier energy, latent rollout, depth 4: `0/8`, h `55.25` |
| `F0_full_score` | compatibility energy, latent rollout, depth 8/16: `0/8`, h `55.375` |
| `F1_full_policy` | verifier energy or remaining-edit, latent rollout, depth 1: `0/8`, h `55.25` |

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
- Listwise verifier-targeted policy prior training covers filled-wrong boards
  with no blanks by adding the positive repair cell to the candidate set.
- Single-CLS compatibility supervision uses binary BCE labels while retaining
  count regression for wrong-count supervision.

## Active: Wide Single-CLS Oracle Probe

Purpose:
- Test whether the one-vector board latent failed because `d_model=256` was too narrow, or because the one-vector geometry/predictor is structurally wrong for Sudoku.
- Keep the readout oracle-only for now: no predicted goal, no waypoint, no value head.

Runs:

| Variant | Train | Eval | Stabilization |
|---|---:|---:|---|
| `W0_ema_vicreg_d1024` | `3817223` running | `3817224` dependency-held | EMA + VICReg, no LDAD |
| `W1_ema_ldad_set_d1024` | `3817225` running | `3817226` dependency-held | EMA + LDAD, no VICReg |
| `W2_ldad_vicreg_set_d1024` | `3817227` running | `3817228` dependency-held | LDAD + VICReg, no EMA |
| `W3_ldad_only_set_d1024` | `3817229` running | `3817230` dependency-held | LDAD only, no EMA/VICReg |

Latest state at 2026-07-06 17:30 CEST:
- All four wide single-CLS train jobs reached the end of 5k training but failed
  while opening `checkpoint-5000.pt` under `/home/atuin/...`: `3815481`,
  `3815483`, `3815485`, and `3815487`.
- Stale dependency-never-satisfied eval jobs `3815482`, `3815484`, `3815486`,
  and `3815488` were canceled.
- Direct write probes originally showed `/home/atuin/c107fa/c107fa12` failing
  new file/directory creation with `EDQUOT` due to the atuin group file quota.
  After deleting `/home/atuin/c107fa/c107fa12/FOMO2` and then
  `/home/atuin/c107fa/c107fa12/python-user-base` on user request, group
  `c107fa` is now at `414,230` files, safely below the `500,000` soft file
  quota, and new file creation under `/home/atuin/.../sequence-editing` works
  again.
- Replacement train/eval pairs were submitted to fresh output root
  `/home/atuin/c107fa/c107fa12/sequence-editing-repair-20260706`.
  Train jobs `3817223`, `3817225`, `3817227`, and `3817229` started immediately
  on `rtxpro6k` node `a2141`; eval jobs `3817224`, `3817226`, `3817228`, and
  `3817230` are normal `afterok` dependencies.
- Current live state: all four replacement train jobs are still running on
  `a2141`; all four eval jobs are still dependency-held. Each run has written
  config/metrics at the repaired output root. Latest logged training progress
  is step `1000/5000` for all four variants, with finite losses and no current
  quota/OOM symptom.
- Expected timing from the previous failed runtimes: W0/W1 training around
  `2026-07-06 21:10-21:25 CEST`, W2/W3 training around
  `2026-07-06 22:40-22:45 CEST`; evals then run individually with an 8h cap, so
  full worst-case results are expected by early `2026-07-07`.

Common setup:
- `latent_representation=single`
- `d_model=1024`, `num_heads=16`, `distance_dim=256`
- 5k steps, effective batch 8 via micro-batch 2 and grad accumulation 4
- counterfactual editable data, K8 smooth/count dense rollout, affected-context dynamics weighting
- eval: oracle raw L2 only, `mpc_beam`, latent rollout and symbolic re-encode, depths `{4,16}`, 8 boards
- repaired output root:
  `/home/atuin/c107fa/c107fa12/sequence-editing-repair-20260706/runs/grid_goal_single_wide`

Storage housekeeping:
- `$HPCVAULT/sequence-editing` was reduced from about `759G` to `5.9G`
- removed redundant `checkpoint-*.pt`, final old `checkpoint.pt`, HF-style
  model/optimizer artifacts, vault cache, and failed zero-byte single-wide files
- after the `/home/atuin` checkpoint failure, additionally removed disposable
  `/home/atuin` vLLM cache, Hugging Face hub cache, and inactive legacy
  `optimizer.pt` files from old sequence-editing runs; this still did not make
  `/home/atuin` writable for new files because the blocking limit is the atuin
  group file quota
- deleted `/home/atuin/c107fa/c107fa12/FOMO2` on user request, freeing about
  `17G` and `8.7k` files; this reduced but did not clear the group file-quota
  overage
- deleted `/home/atuin/c107fa/c107fa12/python-user-base` on user request,
  freeing about `9.8G` and `83k` files; this cleared the atuin group file-quota
  overage and restored new-file creation on `/home/atuin`
- preserved lightweight configs, metrics, diagnostics, and planner/result
  records; backed up the failed W0/W1 metrics/config directories to
  `/scratch/c107fa12_grid_goal_single_wide_failed_w0_w1_20260706_144001`
- canceled stale dependency-never-satisfied evals from the failed wide
  single-CLS submission; replacement train/eval pairs are now running/held

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
