# Runbook

Last updated: 2026-06-12 09:57 CEST

Long-form handoff source of truth: `../sequence-editing-report`.
Deferred planner-ablation notes live in `docs/PLANNER_ABLATION_NOTES.md`.

- Ongoing LaTeX report: `../sequence-editing-report/report.tex`
- Experiment backlog: `../sequence-editing-report/BACKLOG.md`
- Live status: `../sequence-editing-report/STATUS.md`
- Results and insights: `../sequence-editing-report/RESULTS.md`
- Chronological log: `../sequence-editing-report/LOG.md`

## Environment

```bash
source scripts/env.sh
python -m pytest -q tests
```

Runtime outputs default to:

```text
/home/vault/$(id -gn)/$USER/sequence-editing
```

## Active Slurm Snapshot

This table is restricted to jobs submitted from this repo's active puzzle-JEPA
Slurm wrappers. Other user-account HFSA/paired arrays are not part of this
repo snapshot.

Local non-Slurm analysis from 2026-06-11 used the visible A100 with
`scripts/analysis/sudoku_hier_value_probe.py` and the follow-up
`scripts/analysis/sudoku_terminal_projection_probe.py`. Artifacts are:

- `/home/vault/c107fa/c107fa12/sequence-editing/analysis/sudoku_hier_value_terminal_local_20260611.json`
- `/home/vault/c107fa/c107fa12/sequence-editing/analysis/sudoku_hier_value_top_level_20260611.json`
- `/home/vault/c107fa/c107fa12/sequence-editing/analysis/sudoku_hier_value_mcts_root_20260611.json`
- `/home/vault/c107fa/c107fa12/sequence-editing/analysis/sudoku_hier_value_mcts_full_small_20260611.json`
- `/home/vault/c107fa/c107fa12/sequence-editing/analysis/sudoku_terminal_projection_probe_20260611.json`
- `/home/vault/c107fa/c107fa12/sequence-editing/analysis/sudoku_terminal_projection_smoke_20260611.json`

Diagnosis: the state/value heads distinguish very wrong terminal boards but
remain unreliable for one-cell corruptions and local successor ranking;
high-level latent subgoals are not grounded enough for low-level CEM; and
terminal-depth MCTS reaches only depth `2-4` at 256/1024 simulations, with no
terminal leaves. The terminal-projection follow-up gives an upper-bound
diagnostic: even when every first action is completed to a terminal board by
filling the rest with the true solution, Grid 4M `state_value` and
`terminal_energy` still never rank the true solution terminal first globally
across 48 records. Restricting high-level macro actions to an on-manifold
codebook of real 16-step chunks also does not fix selection: the oracle chunk
is never top-1 under the current learned scorers.

Current puzzle-JEPA Slurm state at 2026-06-12 09:57 CEST: no repo-owned puzzle
job is running. Grid 4Q `3715252_[0-11]` is still pending with
`DependencyNeverSatisfied` because Grid 4M task `3711931_0` timed out after
writing useful reset/oracle diagnostics. It is not consuming resources and
partition broadening cannot help; resubmit Grid 4Q without the failed array
dependency if the recursive Grid 4M read is still needed.

Grid 4S/4T macro-action bottleneck/codebook finished cleanly. Training array
`3717328_[0-9]` completed all ten L3 span-4 variants with macro dims
`4/8/16/32`, VQ `(4,64)` and `(8,128)`, plus `state_value` and
`macro_action_advantage` scorer variants. Eval array `3717329_[0-57]`
completed all planner tasks. All checked Grid 4S/4T stderr files are empty.
Result: no Grid 4T planner solved any board. The best macro-bottleneck read is
flat MCTS with oracle `latent_goal`, which gets close but still solves `0/16`
(best mean remaining Hamming `8.56` for terminal-energy `d4_vq64`). Learned
state scores, recursive L1 CEM, and recursive full L2/3 CEM all remain around
`49-52` mean remaining Hamming and solve `0`.

Grid 4U/4V/4W global single-latent MLP also finished cleanly. Grid 4U
`3717900_[0-2]` trained L1/L2/L3 one-token MLP board encoders in `00:13:56`,
`00:24:43`, and `00:25:06`; Grid 4V `3717901_[0-15]` and Grid 4W
`3718124_[0-11]` completed all planner eval tasks. All checked stderr files are
empty. Result: the global bottleneck is a negative control. Learned MPC-CEM
h16/h32/h64 solves `0/64` and sits around `50-52` mean remaining Hamming.
Oracle `latent_goal` MPC-CEM also solves `0/64`, around `44-46` mean remaining
Hamming. Reset/beam oracle `latent_goal` solves `0/128`; the best re-encoded
global read is L3 mean remaining Hamming `19.05`, still far from the old
tokenized oracle reset branch that solved `128/128`.

Grid 4X/4Y mixed correct/wrong rollout training and eval finished cleanly.
Grid 4X `3718216_[0-3]` trained L1 mixed rollout K=2, L1 mixed rollout K=4, L2
mixed hierarchy span-4, and L3 mixed hierarchy span-4. Grid 4Y
`3718217_[0-19]` completed h16/h32 MPC-CEM and recursive hierarchy CEM evals.
All checked stderr files are empty. Result: mixed wrong trajectories did not
rescue the global MLP branch. Learned MPC-CEM solves `0/64`; oracle
`latent_goal` MPC-CEM also solves `0/64`, best mean remaining Hamming `43.73`
for L3 mixed h16. Recursive mixed hierarchy CEM solves `0/24` and remains near
`52` mean remaining Hamming.

Push note: the 2026-06-11 09:15 qualitative-probe updates were committed
locally as `eaf3e14` in this repo and `49ad09b` in
`../sequence-editing-report`. `git push origin main` failed for both with
`ssh: connect to host github.com port 22: Connection timed out` followed by
`fatal: Could not read from remote repository.` The commits are left local.

| Job | State | Notes |
| --- | --- | --- |
| `3664581_[0-1]` | COMPLETED | Grid 0 smoke; infrastructure only, solve remained `0.0`. |
| `3665018_[0-4]` | COMPLETED | Grid 1 one-step/curriculum runs; planning solve remained `0.0`. |
| `3667044_[0-4]` | COMPLETED | Replacement Grid 1 diagnostics completed after earlier cancelled stale diagnostics. |
| `3671344_[0-3]` | COMPLETED | Grid 2A rollout `N=2/4` for Sudoku/Maze; no terminal solve. |
| `3671345_[0-3]` | FAILED | First Grid 2A diagnostics failed on stale CLI flags; superseded by `3673400_[0-3]`. |
| `3673400_[0-3]` | COMPLETED | Replacement Grid 2A diagnostics completed. |
| `3674778_[0-3]` | COMPLETED | Grid 3A training complete; all four roots have `metrics.json` and `checkpoint.pt`. |
| `3674779_[0-3]` | FAILED | First Grid 3A diagnostics failed before model load: wrapper passed comma-separated `--horizons`. |
| `3676904_[0-3]` | COMPLETED | Resubmitted Grid 3A diagnostics completed; all four roots have `diagnostics/diagnostics.json`. |
| `3680019` | COMPLETED | Grid 3B large diagnostics for `sudoku_jepa_5m_local_direct_weighted`; latent rollout solve `0.0`, re-encoded symbolic-state planning solve `1.0`. |
| `3680020` | COMPLETED | Grid 3B `local_direct_weighted` rollout `N=2`; final checkpoint step `5000`, exit `0:0`. |
| `3680021` | COMPLETED | Dependent Grid 3B rollout `N=2` diagnostics; latent terminal-energy solve `4/64`, re-encoded planning `64/64`. |
| `3676879` | COMPLETED | Previous puzzle oversight completed at `2026-05-29 21:48:10 CEST`. |
| `3677391` | COMPLETED | Previous puzzle oversight completed at `2026-05-30 01:43:35 CEST`. |
| `3678050` | COMPLETED | Previous puzzle oversight completed at `2026-05-30 05:48:59 CEST`. |
| `3679094` | COMPLETED | Previous puzzle oversight completed at `2026-05-30 09:44:26 CEST`. |
| `3679877` | CANCELLED | Stale pending oversight cancelled after replacing the prompt with the enhanced template. |
| `3680033` | COMPLETED | Enhanced recurring oversight completed at `2026-05-30 13:41:13 CEST`. |
| `3680652` | COMPLETED | Enhanced recurring oversight completed at `2026-05-30 17:42:01 CEST`. |
| `3681711` | COMPLETED | Enhanced recurring oversight completed at `2026-05-30 21:38:01 CEST`; added/submitted Grid 3C reset-cadence diagnostics. |
| `3682864` | COMPLETED | Enhanced recurring oversight completed at `2026-05-31 01:36:26 CEST`; recorded that Grid 3C was still running. |
| `3682924` | COMPLETED | Grid 3C reset-cadence diagnostics for rollout `N=2`, exit `0:0`; reset every 2/4 solved `64/64` paired boards under step and terminal energy, reset every 8/16 solved `64/64` under terminal energy. |
| `3683472` | COMPLETED | Enhanced recurring oversight completed at `2026-05-31 05:45:27 CEST`, exit `0:0`; submitted successor `3683863`. |
| `3683863` | COMPLETED | Enhanced recurring oversight completed at `2026-05-31 09:36:08 CEST`, exit `0:0`; recorded Grid 3D still running. |
| `3683903` | COMPLETED | Grid 3D reset-large confirmation, exit `0:0`, elapsed `08:25:19`; reset every 4 solved `128/128`, reset every 8 solved `128/128` only under terminal-energy selection. |
| `3684237` | COMPLETED | Enhanced recurring oversight completed at `2026-05-31 13:40:28 CEST`, exit `0:0`; submitted successor `3684889`. |
| `3684889` | NODE_FAIL | Enhanced recurring oversight started at `2026-05-31 17:27:32 CEST` and failed after `00:00:34` on `a0731`; no application stderr, stdout only job statistics. |
| `3687722` | COMPLETED | Replacement enhanced recurring oversight ran `2026-06-01 12:57:00-13:21:50 CEST`, exit `0:0`; submitted successor `3688542`. |
| `3688542` | COMPLETED | Enhanced recurring oversight ran `2026-06-01 16:57:23-17:12:45 CEST`, exit `0:0`; submitted successor `3689344` and queued Grid 4A diagnostics. |
| `3689344` | CANCELLED | Later oversight from `3688542` was cancelled before start at `2026-06-01 18:29:32 CEST`; replaced by `3689685`. |
| `3689685` | CANCELLED | Replacement oversight was cancelled before start at `2026-06-01 19:57:58 CEST`; replaced by `3691526`. |
| `3691526` | COMPLETED | Final enhanced recurring oversight ran `2026-06-02 14:22:40-14:32:29 CEST`, exit `0:0`; submitted successor `3692215`. |
| `3692215` | CANCELLED | Cancelled by user request at `2026-06-02 14:40:41 CEST`; recurring oversight is now disabled. |
| `3688587_[0-2]` | CANCELLED | User-approved cancellation at `2026-06-01 14:46:56 CEST`; pre-HWM-correction Grid 4A baseline jobs ran `01:40:56` and wrote step-1 metrics. |
| `3688921_[0-2]` | CANCELLED | Superseded after the user requested exact report-style hierarchical planning; cancelled at `2026-06-01 15:01:20 CEST` after `00:14:07` on `a0531`, `a0731`, and `a0931`; no checkpoints. |
| `3688986_[0-2]` | COMPLETED | Exact-recipe Grid 4A training completed cleanly on 2026-06-01; all three roots have final `checkpoint.pt` at step 5000. |
| `3689396_[0-2]` | COMPLETED | Grid 4A learned-energy CEM diagnostics completed; solve `0/64` for L1/L2/L3. |
| `3689397_[0-1]` | COMPLETED | Grid 4A report-style subgoal CEM diagnostics completed; solve `0/32` for L2/L3. |
| `3691590_[0-2]` | COMPLETED | Grid 4B learned-energy reset/beam diagnostic for L1/L2/L3; exit `0:0`; solved `0/128` for all three levels. |
| `3695040` | COMPLETED | Grid 4C L1 oracle reset/calibration sanity; exit `0:0`, elapsed `05:16:51`; reset every 4 and re-encoded oracle-goal planning solved `128/128`. |
| `3696588_[0-5]` | FAILED | First Grid 4D submission failed immediately before training: Hydra rejected new `training.*` keys without `+` override syntax. |
| `3696609_[0-5]` | FAILED | Second Grid 4D submission fixed Hydra overrides but failed before checkpointing: oversized auxiliary contrastive load (`512` examples x `16` negatives) caused OOM on five tasks; one task hit stale HF cache file handle. |
| `3696616_[0-5]` | COMPLETED | Grid 4D L1 contrastive goal-energy ablation; learned-energy reset/beam solved `0/128` for every variant. |
| `3698281_[0-6]` | COMPLETED | Grid 4E action-candidate rank analysis; original L1 top1 `0.040`, best contrastive top1 `0.049`. |
| `3702008` | COMPLETED | Grid 4I replacement diagnostics-only job. Learned discounted-reachability reset/beam solved `0/128`; oracle latent-goal reset control preserved dynamics with reset every 4 and re-encoded planning `128/128`. |
| `3702066` | COMPLETED | Grid 4J original L1 energy-action calibration; mean all-action absolute error `0.00443`, mean local Pearson `0.337`. |
| `3702254_[0-1]` | COMPLETED | Grid 4K ListNet learned-energy ranking. Learned-score reset/beam solved `0/128` for both label variants; oracle reset control solved `128/128` for remaining-wrong relevance and `112/128` for latent-goal relevance. |
| `3705899_[0-5]` | COMPLETED | Grid 4L scorer-spread L1 ablation first six variants. Every learned-score reset/beam variant solved `0/128`; every oracle latent-goal reset control solved `128/128`. |
| `3705899_6` | TIMEOUT | Grid 4L MuZero-like value+MCTS. Training plus normal learned-score/oracle diagnostics completed; extra MCTS diagnostic timed out. Learned reset/beam solved `0/128`, oracle reset control solved `128/128`. |
| `3705900` | COMPLETED | Fixed-sign Grid 4I diagnostic rerun using `--planning-score goal_value`; solved `0/128`, terminal rate `0.172`, mean remaining Hamming `49.83`. |
| `3711931_[0-3]` | MIXED | Grid 4M L3 span-4 hierarchical value ablation. Task `_0` timed out after `1-00:00:26`, but tasks `_1`-`_3` completed cleanly. Useful reset/oracle diagnostics exist: learned-score reset solves `0/128` for every variant, while oracle latent-goal reset/calibration solves `128/128` for all four. The timeout caused Grid 4Q's `afterok` dependency to become unsatisfiable. |
| `3711983` | COMPLETED | Grid 4N true macro-action advantage L3 span-4 completed at 2026-06-11 06:45:29 CEST, exit `0:0`. Oracle latent-goal reset/calibration: no-reset terminal-energy selection solved `127/128` with mean remaining Hamming `0.008`, and reset every 4 plus re-encoded planning solved `128/128`. Latent-goal subgoal CEM solved `0/32`, terminal `0/32`, mean remaining Hamming `48.47`; learned macro-action top-score subgoal CEM also solved `0/32`, terminal `0/32`, mean remaining Hamming `49.72`, so the learned top score is not directionally useful in this read. |
| `3714062_[0-3]` | TIMEOUT | Grid 4O inference-only MCTS diagnostics timed out at 2026-06-09 04:23:43 CEST after 8h. No `diagnostics_mcts_*` directories or MCTS JSON/JSONL artifacts were written because records are emitted only after full completion. |
| `3715249_[0-3]` | COMPLETED | Grid 4P smaller streaming MCTS diagnostic on original L1. All four tasks completed at 2026-06-10 11:53-11:54 CEST, exit `0:0`. Learned `goal_energy` d4/d8 solved `0/32`, terminal `0`, mean remaining Hamming `47.78`/`48.72`; oracle `latent_goal` d4/d8 solved `0/32`, terminal `0`, mean remaining Hamming `9.88`/`10.03`. Root debug top-1 goal-value writes: learned `65/438`, `70/452`; oracle `376/449`, `360/440`. |
| `3715252_[0-11]` | PENDING/NEVER | Grid 4Q recursive hierarchy diagnostics on Grid 4M checkpoints. Dependency `afterok:3711931` is `DependencyNeverSatisfied` because Grid 4M task `_0` timed out. No recursive Grid 4Q artifacts exist. Resubmit without the failed array dependency if this read is still desired. |
| `3715251_[0-2]` | COMPLETED | Grid 4R recursive hierarchy diagnostics on Grid 4N macro-action checkpoint completed after Grid 4N. All three tasks exited `0:0` with empty stderr. Recursive macro-action top score solved `0/16` and terminal `0/16` for `cem`, `gd`, and `gd_reachability`; mean remaining Hamming was `51.81`, `52.94`, and `51.31` respectively. Learned macro-action top score is not directionally useful here and is worse than the non-recursive latent-goal subgoal CEM control. |
| `3717328_[0-9]` | COMPLETED | Grid 4S HWM-style L3 span-4 macro-action bottleneck/codebook training. All ten tasks completed cleanly in `05:13:05`-`05:42:25`; stderr files are empty. |
| `3717329_[0-57]` | COMPLETED | Grid 4T eval-only planner matrix for Grid 4S. All tasks completed cleanly, stderrs empty. Every planner read solved `0`; best oracle MCTS mean remaining Hamming was `8.56`, while recursive CEM reads stayed near `50-52`. |
| `3717900_[0-2]` | COMPLETED | Grid 4U global single-latent MLP JEPA training in `scripts/slurm/run_grid4u_global_mlp_latent.slurm`. L1/L2/L3 completed cleanly in `00:13:56`, `00:24:43`, and `00:25:06`; stderr files are empty. |
| `3717901_[0-15]` | COMPLETED | Grid 4V eval-only planner matrix for Grid 4U. All tasks completed cleanly, stderrs empty. h16 MPC-CEM and reset/beam solve `0`; oracle `latent_goal` is also poor. |
| `3718124_[0-11]` | COMPLETED | Grid 4W eval-only long-horizon MPC-CEM for Grid 4U. All tasks completed cleanly, stderrs empty. h32/h64 both solve `0/64` under learned and oracle scores. |
| `3718216_[0-3]` | COMPLETED | Grid 4X mixed correct/wrong rollout training in `scripts/slurm/run_grid4x_global_mlp_mixed_rollout.slurm`. All four tasks completed cleanly in `00:13:12`-`00:20:57`; stderr files are empty. |
| `3718217_[0-19]` | COMPLETED | Grid 4Y dependent mixed-rollout planner matrix. All tasks completed cleanly, stderrs empty. h16/h32 MPC-CEM and recursive hierarchy CEM solve `0` across the board. |
| `3715253`; `3715250`, `3715254`, `3715255` | CANCELLED | User-requested one-shot Grid 4P/4Q/4R oversight jobs all began together at 2026-06-10 11:42:38 CEST. Stale duplicate active watch jobs `3715250`, `3715254`, and `3715255` were cancelled at 11:44:50 CEST with logs preserved; stale running watch `3715253` was cancelled at 11:56:08 after it cancelled the first new scheduled attempt. |
| `3715429`, `3715430`, `3715431`, `3715433`, `3715432` | CANCELLED | First begin-time-blocked attempt for the user-requested 2026-06-10 18:00/20:00 and 2026-06-11 00:00/04:00/08:00 CEST checks. Cancelled before start at 11:53:40 CEST by stale watch `3715253`; superseded by `3715446`-`3715450`. |
| `3715446`; `3715447`; `3715448`; `3715449`; `3715450` | COMPLETED | Exact-time one-shot oversight `3715446` completed at 2026-06-10 18:18:46 CEST, `3715447` at 20:12:29 CEST, `3715448` at 2026-06-11 00:14:48 CEST, `3715449` at 04:17:02 CEST, and `3715450` at 08:24:05 CEST after starting on A40 node `a1621` at 08:00:26. The final check confirmed proxy inheritance and did not submit a successor oversight job. |
| `3714106` | COMPLETED | First user-requested one-shot Grid 4O oversight completed at 2026-06-08 22:45:38 CEST, exit `0:0`. Non-recurring; do not submit successors. |
| `3714107` | COMPLETED | Second user-requested one-shot Grid 4O oversight completed at 2026-06-09 00:43:12 CEST, exit `0:0`. Non-recurring; do not submit successors. |
| `3714108` | COMPLETED | Third and final user-requested one-shot Grid 4O oversight completed at 2026-06-09 02:47:39 CEST. Non-recurring; do not submit successors. |

Push note: the 2026-06-09 00:36 CEST oversight docs were committed locally in
both repos, but `git push` failed for both with `ssh: connect to host github.com
port 22: Connection timed out` followed by `fatal: Could not read from remote
repository.` The 2026-06-09 13:10 CEST updates were also committed locally
(`a62fd94` in this repo, `12f5114` in `../sequence-editing-report`), and both
pushes failed with the same GitHub SSH timeout and fatal remote-read error.
The 2026-06-10 18:03 CEST handoff updates were committed locally in both repos;
`git push origin main` failed for both with `ssh: connect to host github.com
port 22: Connection timed out` followed by `fatal: Could not read from remote
repository.` A manual retry from the login shell at 19:22 CEST succeeded for
both repos (`a47d190` in this repo and `9d8bf84` in
`../sequence-editing-report`). Future exact-time oversight prompts now
explicitly verify `http_proxy`/`https_proxy` inheritance and distinguish Codex
connectivity from the known GitHub SSH port-22 egress failure.
The 2026-06-10 20:04 CEST handoff updates were committed locally in both repos
(`f899543` in this repo and `16ee2a7` in `../sequence-editing-report`), and
both pushes failed with the same exact GitHub SSH error:
`ssh: connect to host github.com port 22: Connection timed out` followed by
`fatal: Could not read from remote repository.` The commits are left local; no
replacement oversight job was submitted for the push failure.
The 2026-06-11 00:04 CEST handoff updates were committed locally in both repos
(`3e61310` in this repo and `8f6729e` in `../sequence-editing-report`), and
both pushes failed with the same exact GitHub SSH error:
`ssh: connect to host github.com port 22: Connection timed out` followed by
`fatal: Could not read from remote repository.` The commits are left local; no
replacement oversight job was submitted for the push failure.
The 2026-06-11 04:04 CEST handoff updates were committed locally in both repos
(`fe32ce4` in this repo and `4b71b27` in `../sequence-editing-report`), and
both pushes failed with the same exact GitHub SSH error:
`ssh: connect to host github.com port 22: Connection timed out` followed by
`fatal: Could not read from remote repository.` The commits are left local; no
replacement oversight job was submitted for the push failure.
The 2026-06-11 08:05 CEST handoff updates were committed locally in both repos
(`4e963e2` in this repo and `6bb9299` in `../sequence-editing-report`), and
both pushes failed with the same exact GitHub SSH error:
`ssh: connect to host github.com port 22: Connection timed out` followed by
`fatal: Could not read from remote repository.` The commits are left local; no
replacement oversight job was submitted for the push failure.
The 2026-06-11 08:17 CEST terminal-subgoal handoff update was committed locally
in both repos (`cdebff9` in this repo and `83486c3` in
`../sequence-editing-report`), and both pushes failed with the same exact
GitHub SSH error: `ssh: connect to host github.com port 22: Connection timed
out` followed by `fatal: Could not read from remote repository.` The commits
are left local; no replacement oversight job was submitted for the push
failure.

Check live state:

```bash
squeue -j 3711931,3715252,3717328,3717329 -o "%.18i %.9T %.28j %.10M %.20S %R"
sacct -j 3714106,3714107,3714108,3714062,3711931,3711983,3705899,3705900,3702254,3702008,3702066,3699523,3698893,3698394,3698281,3696616 --format=JobID,JobName%30,State,ExitCode,Elapsed,Start,End,NodeList
```

## Current Operational Read

Grid 3B rollout `N=2` improved proximity, but it did not satisfy the pure
latent exact-solve gate. Job `3680021` reports latent terminal-energy planning
solve `4/64`, filled-board terminal rate `26/64`, and mean remaining Hamming
`2.453`. Re-encoded symbolic-state planning still solves `64/64`.

Compared with the lead large diagnostics (`3680019`), rollout `N=2` cuts
terminal-energy mismatches from 299 to 157 and reduces drift at 10/20 oracle
steps from `0.079/1.742` to `0.041/1.495`. Terminal weighted latent drift is
not fixed (`2.014 -> 2.163`), so the main interpretation still holds: under
oracle-goal diagnostics, the action scorer is sufficient when candidate boards
are re-encoded, while stale latent rollout remains the bottleneck. The result
is not deployable because it still uses the oracle goal and because latent and
re-encoded planner samples are not paired example-by-example.

Grid 3C reset-cadence diagnostics (`3682924`) passed the mechanism gate, and
Grid 3D reset-large confirmation (`3683903`) confirmed it on a larger paired
sample. On 128 paired boards, no-reset terminal-energy planning solved `7/128`
with mean remaining Hamming `2.398`, reset every 4 solved `128/128` under both
step- and terminal-energy selection, reset every 8 solved `91/128` under
step-energy but `128/128` under terminal-energy selection, and full re-encoded
planning solved `128/128`.

Planning intentionally allows overwrites/conflicts on mutable Sudoku cells for
diagnosis, so terminal boards can be fully filled but wrong. Treat online
H1/H2/H4, terminal fill rate, and oracle-goal reset results as diagnostics, not
deployable solve quality. Generated analysis artifacts live under
`../sequence-editing-report/assets/grid3b/`, including the new Grid 3D
reset-large planning plots, CSVs, and concrete paired examples. The decision
gate is now satisfied for implementing a planner-state reset/re-encoding
branch that keeps symbolic candidate boards as the state of record and
re-encodes latents every 4 actions.

Oversight chain issue: successor oversight `3684889` failed with `NODE_FAIL`
after 34 seconds on `a0731`; there was no application stderr. Replacement
oversight `3687722` completed, oversight `3688542` completed cleanly at
`2026-06-01 17:12:45 CEST`, its successor `3689344` was later cancelled before
start, replacement `3689685` was also cancelled before start, and replacement
oversight `3691526` completed at `2026-06-02 14:32:29 CEST`; its successor
`3692215` was cancelled by user request at `2026-06-02 14:40:41 CEST`.
Other visible HFSA/paired user-account arrays are outside this repo snapshot.
Partition housekeeping at 17:00 CEST: `sinfo` showed idle `a100`, `a40`, and
`rtxpro6k` nodes, but at that time the only pending repo jobs were
dependency-blocked diagnostics or begin-time-blocked oversight, so no
`scontrol update` was useful.

Grid 4A check at 14:47 CEST: pre-HWM-correction jobs `3688587_[0-2]` were
cancelled after user approval. Corrected HWM-style jobs `3688921_[0-2]` started
immediately, but were later superseded by the exact report-style planner request.
At submission time the corrected HWM run roots did not yet contain metrics or
checkpoints; CEM diagnostics should wait for `checkpoint.pt`.

Grid 4A check at 15:10 CEST: `3688921_[0-2]` was cancelled after the user asked
for the exact report-style planner rather than the intermediate primitive-candidate
hierarchical score path. Replacement exact-recipe training is `3688986_[0-2]`,
running since `2026-06-01 15:09:33 CEST`. The corrected run roots currently
contain refreshed `config.json` files but no metrics or checkpoints yet.

Grid 4A check at 17:00 CEST: `3688986_[0-2]` is still running. All three roots
have `checkpoint-1000.pt` and `checkpoint.pt`. Step-1000 online metrics are:
L1 eval loss `0.000378`, goal-energy MSE `0.00341`, rank `12.66`; L2 eval loss
`0.000430`, goal-energy MSE `0.000676`, rank `13.16`; L3 eval loss `0.000404`,
goal-energy MSE `0.00125`, rank `11.19`. H1/H2/H4 online solve is `1.0` for all
three, but this remains a small sanity metric. Learned-energy CEM diagnostics
are queued as `3689396_[0-2]` with `afterok:3688986`; report-style subgoal CEM
diagnostics are queued as `3689397_[0-1]` with `afterok:3689396`. Generated
step-1000 artifacts live in `../sequence-editing-report/assets/grid4a/`.

Grid 4A check at 18:35 CEST: `3688986_[0-2]` is still running. L1 has reached
step 3000 with eval loss `0.000171` and goal-energy MSE `0.000335`; L2 has
reached step 2000 with eval loss `0.000257`, goal-energy MSE `0.000431`, and
hierarchy loss `0.0158`; L3 has reached step 2000 with eval loss `0.000243`,
goal-energy MSE `0.000182`, and hierarchy loss `0.0132`. H1/H2/H4 online solve
remains `1.0` for all three, still only a sanity metric. The queued diagnostics
remain dependency-blocked.

Grid 4A result check at 10:23 CEST on 2026-06-02: training `3688986_[0-2]`,
learned-energy CEM `3689396_[0-2]`, and report-style subgoal CEM
`3689397_[0-1]` all completed cleanly. Training losses are healthy at step 5000
and online H1/H2/H4 remains `1.0`, but the actual CEM gates failed. Learned
goal-energy CEM solved `0/64` for every level, with terminal rate `0.0` and
mean remaining Hamming L1/L2/L3 `50.80`/`50.33`/`49.70`. Report-style subgoal
CEM solved `0/32` for L2 and L3, with mean remaining Hamming `48.31` and
`49.28`; L3 produced one terminal but wrong board. Artifacts are in
`../sequence-editing-report/assets/grid4a/`.

Grid 4B submission at 10:49 CEST on 2026-06-02: implemented and submitted
`3691590_[0-2]` via `scripts/slurm/run_grid4a_goal_energy_reset_diagnostics.slurm`.
It tests beam search over legal Sudoku writes with `--planning-score goal_energy`,
`--planning-beam-size 4`, `--planning-branch-size 8`, and `--reset-cadences 4`
on the three exact-recipe Grid 4A checkpoints. Output roots are
`$PUZZLE_JEPA_WORK_ROOT/runs/sudoku_jepa_5m_goal_energy_hwm_{l1,l2_span9,l3_span3}/diagnostics_reset_goal_energy`.

Grid 4B live check at 14:25 CEST on 2026-06-02: `3691590_[0-2]` is still
running after about `03:34` on `a0532`, `a0537`, and `a0731`. Stderr is empty
and stdout contains only Slurm prologues. No `diagnostics_reset_goal_energy`
directory exists yet; this is not by itself a failure because
`puzzle_jepa.eval.diagnostics` writes output only after finishing latent,
re-encoded, and paired reset planning. `sstat` shows active CPU time and max RSS
about `1.6-1.7 GiB` for observed tasks.

Grid 4B result check at 09:24 CEST on 2026-06-03: `3691590_[0-2]` completed
cleanly on 2026-06-02 with empty stderr. Learned-energy beam/reset solved
`0/128` for L1/L2/L3. Paired reset `reset_every_4`, no-reset, and re-encoded
variants all match because `--planning-score goal_energy` scores symbolic
candidate boards directly, so latent reset cannot repair a bad learned energy
ranking. Mean remaining Hamming for paired reset is L1 `47.41`, L2 `46.23`,
L3 `45.84`, with terminal rate `0.0`. Results live in
`diagnostics_reset_goal_energy`; summary CSV:
`../sequence-editing-report/assets/grid4a/grid4b_reset_goal_energy_summary.csv`.

Grid 4C submission at 11:11 CEST on 2026-06-03: added trajectory calibration
records/plots to `puzzle_jepa.eval.diagnostics` and submitted `3695040` via
`scripts/slurm/run_grid4c_l1_oracle_reset_calibration.slurm`. This is the quick
sanity check requested after Grid 4B: same L1 checkpoint as `3691590_0`, same
beam/reset method that solved the oracle-goal Grid 3D control, but with
`--planning-score latent_goal` instead of learned goal energy. It writes
`diagnostics_reset_oracle_calibration`, including
`goal_energy_calibration_records.jsonl`, `goal_energy_abs_error_by_step.png`,
and `goal_energy_example_*.png`.

Grid 4C result check at 17:53 CEST on 2026-06-03: `3695040` completed cleanly
with exit `0:0`. Reset every 4 and re-encoded oracle-goal planning solved
`128/128`; no-reset latent terminal-energy planning solved `79/128` and
no-reset step-energy planning solved `26/128`. Learned energy calibration on
reset-every-4 terminal trajectories has mean absolute error about `0.010`,
predicted monotone-nonincreasing rate about `0.923`, true latent-distance
monotone rate `1.0`, initial predicted/true means `0.213/0.214`, and final
predicted/true means `0.00023/0.00002`. Interpretation: checkpoint dynamics are
not the issue; the learned goal-energy scorer is close globally but too noisy
locally to replace oracle latent MSE as an action-selection objective.

Oversight cancellation at 14:41 CEST: by user request, pending successor
`3692215` was cancelled and the recurring oversight wrapper/prompt were removed.
Do not schedule further `puzzle_oversight` jobs.

Implementation correction at 14:03 CEST: the user clarified that the hierarchy
should have an explicit higher-level action encoder over the lower-level action
span and that K should be configurable. The code now adds
`ActionSequenceEncoder`, `hierarchy_span`, recursive higher-level action
encoding, and a diagnostic `hierarchical_latent_goal` CEM score path. Corrected
run roots are `sudoku_jepa_5m_goal_energy_hwm_l1`,
`sudoku_jepa_5m_goal_energy_hwm_l2_span9`, and
`sudoku_jepa_5m_goal_energy_hwm_l3_span3`, so they do not overwrite the
cancelled pre-correction roots.

Exact report-style planning implemented at 15:10 CEST: diagnostics now have a
`hierarchical_subgoal_cem` path. High-level CEM samples continuous latent
macro-action sequences, rolls out the higher-level predictor toward the solved
board latent, takes the first predicted high-level latent as the subgoal, and
then runs low-level categorical CEM over primitive Sudoku writes to reach that
subgoal. The high-level latent action is not decoded into primitive actions.
That subgoal diagnostic completed as `3689397_[0-1]`. The older
`run_grid4a_hierarchical_cem_diagnostics.slurm` is only a comparison diagnostic
that scores primitive candidate chunks through the action encoder; it is not the
exact report recipe.

Focused validation passed under the repo venv:

```bash
source scripts/env.sh
pytest tests/test_puzzle_models.py tests/test_puzzle_hydra.py -q
```

The 13:06 CEST oversight rerun of that pytest command was interrupted before
collection because Python imports were stuck in shared filesystem waits
(`rpc_wait_bit_killable`/`folio_wait_bit_common`). The Slurm wrapper syntax
checks and `py_compile` for the changed Grid 4A modules passed.

Grid 4A and Grid 4B diagnostics completed cleanly but failed the solve gates.
Grid 4B isolates the issue: learned goal energy does not rank useful symbolic
states even under beam/reset, so fix scorer/ranking/calibration or add a
verifier/goal objective before changing CEM action parameterization or trying
larger models.

For the older primitive-candidate hierarchy comparison diagnostic, run
`scripts/slurm/run_grid4a_hierarchical_cem_diagnostics.slurm` only after the
exact subgoal planner is recorded or if a direct comparison is needed.

Recurring oversight is disabled by user request as of 2026-06-02 14:41 CEST.
Do not schedule further `puzzle_oversight` jobs. The next safe step is to debug
or replace the learned goal-energy scorer, using Grid 4C calibration records and
the Grid 3D/Grid 4C oracle-goal reset controls as baselines.
