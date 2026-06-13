# Runbook

Last updated: 2026-06-13 14:11 CEST

Long-form handoff source of truth: `../sequence-editing-report`.

- Ongoing LaTeX report: `../sequence-editing-report/report.tex`
- Experiment backlog: `../sequence-editing-report/BACKLOG.md`
- Clean Grid5 backlog: `../sequence-editing-report/GRID5_BACKLOG.md`
- Clean Grid5 plan: `../sequence-editing-report/GRID5_PLAN.md`
- Clean Grid5 log: `../sequence-editing-report/GRID5_LOG.md`
- Live status: `../sequence-editing-report/STATUS.md`
- Results and insights: `../sequence-editing-report/RESULTS.md`
- Chronological log: `../sequence-editing-report/LOG.md`

## Active Surface

The active experiment surface has moved to Grid 6 causal trajectory JEPA. Grid
5 remains the negative compact single-state baseline/background.

- Config: `configs/puzzle/grid6_sudoku_trajectory.yaml`
- Model: `puzzle_jepa/models/trajectory_jepa.py`
- Train: `puzzle_jepa/train/grid6.py`
- Planner eval: `puzzle_jepa/eval/grid6_planner_matrix.py`
- Slurm train: `scripts/slurm/run_grid6_causal_trajectory.slurm`
- Slurm eval: `scripts/slurm/run_grid6_causal_trajectory_eval.slurm`
- Grid5 background model/probes:
  `puzzle_jepa/models/sigreg_jepa.py`,
  `scripts/analysis/grid5_geometry_probe.py`

Old `grid0`-`grid4` experiment configs and Slurm wrappers were removed from the
active tree. Historical results remain in `../sequence-editing-report`.

## Environment

```bash
source scripts/env.sh
pytest tests/test_grid6_trajectory.py tests/test_puzzle_hydra.py -q
```

Runtime outputs default to:

```text
/home/vault/$(id -gn)/$USER/sequence-editing
```

## Slurm Snapshot

Grid 6 causal trajectory JEPA was submitted at 2026-06-13 14:11 CEST.

- Train array: `3739195_[0-1]`
- Dependent planner eval array: `3739196_[0-1]`, dependency `afterok:3739195`
- Slurm state at 2026-06-13 14:13 CEST: train task `3739195_0` running on
  `a40` node `a0228`; train task `3739195_1` running on `a100` node `a0804`;
  eval array `3739196_[0-1]` pending with reason `Dependency`
- Partition request: `a40,a100,rtxpro6k`
- Resources: train one GPU, 8 CPUs, 18h; eval one GPU, 4 CPUs, 12h
- Run roots:
  `$PUZZLE_JEPA_WORK_ROOT/runs/grid6_causal_traj_k1_d320` and
  `$PUZZLE_JEPA_WORK_ROOT/runs/grid6_causal_traj_mh_d320`
- Architecture: causal board/action history encoder, frozen EMA target
  encoder, action-chunk encoder, horizon predictor, JEPA latent MSE + SIGReg +
  learned `goal_energy` head
- Scale: `23.25M` total params including frozen EMA target encoder, `15.70M`
  trainable params
- Tasks:
  - task `0`: one-step horizons `[1]`
  - task `1`: multi-horizon `[1,2,4,8,16]`
- Eval matrix after training: Beam/CEM/MCTS, `symbolic_reencode` vs
  `latent_rollout`, oracle `latent_goal` vs learned `goal_energy`, horizons
  `4/8/16` plus mean-prefix score over `1/2/4/8/16`.
- Verification before submission: `py_compile`, `bash -n`, focused Grid6
  pytest, combined Grid6+Hydra pytest, one-step train smoke, and planner CLI
  smoke.

Grid 5 was submitted as `3722613_[0-23]` at 2026-06-12 11:29 CEST and has
completed.

- Partition request: `a40,a100,rtxpro6k`
- Resource request: one GPU per task, 8 CPUs, 8h wall time
- Final state: all 24 tasks completed with exit code `0:0`
- Runtime: about 10-12 minutes on `rtxpro6k`, 20-27 minutes on `a40`
- Stderr: all Grid 5 stderr files are empty
- Output roots:
  `$PUZZLE_JEPA_WORK_ROOT/runs/grid5_sigreg_{encoder}_{predictor}_{state|delta}_z{32|64|128}`

Legacy Grid 4Z `3722524` completed cleanly with exit `0:0` after `03:52:04`.
It is superseded by Grid 5 and failed the planner gate: recursive hierarchy CEM
with `latent_goal` and `goal_energy` both solved `0/16`, terminal rate `0.0`,
mean remaining Hamming `50.5625`. Do not extend Grid 4 unless explicitly
requested. Grid 4Q `3715252_[0-11]` remains pending with
`DependencyNeverSatisfied`; it is not consuming resources.

Grid 5 posthoc MPC-CEM lookahead diagnostics were submitted as
`3724325_[0-23]` at 2026-06-12 13:44 CEST and completed cleanly.

- Wrapper: `scripts/slurm/run_grid5_mpc_cem_diagnostics.slurm`
- Eval module: `puzzle_jepa/eval/grid5_mpc_cem_diagnostics.py`
- Final state: all 24 tasks completed with exit `0:0`
- Purpose: LeWorldModel-style MPC-CEM over horizons `4/8/16/32/64`
- Outputs:
  `$PUZZLE_JEPA_WORK_ROOT/runs/grid5_sigreg_*/diagnostics_mpc_cem/`

Grid 5 recursive rollout training was submitted as `3724413_[0-5]` at
2026-06-12 14:00 CEST and completed cleanly.

- Wrapper: `scripts/slurm/run_grid5_recursive_rollout.slurm`
- Final state: all 6 tasks completed with exit `0:0`
- Fixed base: MLP encoder, delta prediction, latent size `128`
- Factors: predictor `mlp|ar_transformer` x recursive rollout K `2|4|8`
- Outputs:
  `$PUZZLE_JEPA_WORK_ROOT/runs/grid5_recursive_mlp_{predictor}_delta_z128_k{K}`
- Each task trains, runs standard Grid 5 diagnostics, then runs MPC-CEM
  horizons `4/8/16/32/64` with oracle `latent_goal` and learned `goal_energy`.

Grid 5 recursive rollout full-state counterpart was submitted as
`3724500_[0-5]` at 2026-06-12 14:14 CEST and completed cleanly.

- Wrapper: `scripts/slurm/run_grid5_recursive_rollout_state.slurm`
- Final state: all 6 tasks completed with exit `0:0`
- Same matrix as `3724413`, but `model.predict_delta=false`
- Outputs:
  `$PUZZLE_JEPA_WORK_ROOT/runs/grid5_recursive_mlp_{predictor}_state_z128_k{K}`

Grid 5B 10M stabilizer screen was submitted as `3724634_[0-11]` at
2026-06-12 15:54 CEST. Tasks `0-5` hit Slurm `NODE_FAIL` on node `a2143`
after about four minutes with empty stderr. Original tasks `6-11` completed
cleanly. The failed slice was resubmitted as `3724689_[0-5]` with
`--exclude=a2143` and completed cleanly by 2026-06-12 17:12 CEST.

- Wrapper: `scripts/slurm/run_grid5b_10m_stabilizer_screen.slurm`
- Partition request: `a40,a100,rtxpro6k`
- Resource request: one GPU per task, 8 CPUs, 12h wall time
- Run roots: `$PUZZLE_JEPA_WORK_ROOT/runs/grid5b_10m_*`
- Stderr check for failed original tasks `0-5`: empty; Slurm reason was
  `NODE_FAIL`, not a Python traceback
- Final Grid5B stderrs checked for rerun `0-5` and original `6-11`: empty
- Trainable params: `10.6M` to `13.4M`; EMA variants carry frozen target
  encoders, so total params are larger but trainable params stay in this range

The 12-job screen covers:

- stabilizer: SIGReg, EMA target + SIGReg, VICReg, EMA target + VICReg
- rollout loss: K=1 vs K=4
- prediction target: full-state vs delta
- architecture: MLP vs CLS-transformer encoder, MLP vs AR-transformer predictor

Each task trained, then ran standard diagnostics, predicted-latent MPC-CEM, and
symbolic re-encode MPC-CEM. Grid5B improved proximity but still did not solve:
best symbolic oracle read is `grid5b_10m_canonical_ema_vicreg_k4`, h8 mean
remaining Hamming `41.00`, root goal-value rate `0.500`, solve `0/4`.
Predicted-latent MPC-CEM solved `0` for every variant; best proximity is
`grid5b_10m_canonical_ema_sigreg_k4`, h64 `goal_energy`, mean remaining
Hamming `49.50`.

Grid 5C planner matrix was added as
`scripts/slurm/run_grid5c_planner_matrix_eval.slurm` and
`puzzle_jepa/eval/grid5_planner_matrix.py`.

- Verification passed: `py_compile`, `bash -n`, full
  `pytest tests/test_grid5_sigreg.py -q`, and a tiny real-checkpoint CLI smoke.
- Planner optimizers: `beam`, `mcts`, and continuous action-embedding
  nearest-neighbor CEM (`nn_cem`).
- Transition axis: exact symbolic board application + re-encode at the horizon
  (`symbolic_reencode`) vs recursive latent predictor rollout
  (`latent_rollout`).
- Scoring axis: oracle solved-board latent distance (`latent_goal`) vs learned
  terminal-energy head (`goal_energy`).
- Action mode: mutable-cell overwrites, preserving clue cells.
- Submitted eval jobs:
  - `3724691_[0-5]`, dependent on Grid 5B rerun `3724689`; started at
    2026-06-12 17:13 CEST on `a40`
  - `3724698_[9-11]`, started immediately for already-completed tasks `9-11`
    but landed on node `a2143`; monitor for repeat node failure
  - `3724700_[6]`, `3724701_[7]`, `3724702_[8]`, each dependent on the
    matching original Grid 5B task; tasks `6-8` have started
- Grid5C tasks `3724698_[9-11]`, `3724700_6`, `3724701_7`, and `3724702_8`
  timed out before writing summaries. Tasks `3724691_[0-5]` also timed out at
  2026-06-13 05:13 CEST. All checked stderrs contain only Slurm time-limit
  messages; no Python traceback was found.
- `puzzle_jepa/eval/grid5_planner_matrix.py` now writes
  `planner_records.jsonl` and `planner_summary.json` incrementally after each
  completed mode, so future timeouts preserve partial reads.
- Small streaming replacement probe `3728790` completed cleanly in `01:03:08`
  on a40 node `a0124`. It evaluated `grid5b_10m_canonical_ema_vicreg_k4` on
  one board at h8 across `beam|mcts|nn_cem`, `symbolic_reencode|latent_rollout`,
  and `latent_goal|goal_energy`. Best result was MCTS +
  `symbolic_reencode` + oracle `latent_goal`, remaining Hamming `37` from
  start `55`, solve `0/1`. Beam oracle symbolic was `39`; latent-rollout modes
  stayed `53-55`; learned energy stayed weak (`49-54`). Output root:
  `$PUZZLE_JEPA_WORK_ROOT/runs/grid5b_10m_canonical_ema_vicreg_k4/diagnostics_planner_matrix_probe_20260613/`.

Grid 5 geometry probe was added as `scripts/analysis/grid5_geometry_probe.py`
and run locally on `grid5b_10m_canonical_ema_vicreg_k4`. Artifact:
`$PUZZLE_JEPA_WORK_ROOT/analysis/grid5_geometry_probe_canonical_ema_vicreg_k4_20260613/`.
Key read: learned `goal_energy` true-terminal top1 is `0/16` among one-cell
corrupt terminal boards; latent/Hamming nearest-neighbor Spearman is `0.133`;
best wrong action displacement beats gold goal-direction cosine in `84.4%` of
sampled states. This puts Grid5 on the oracle-symbolic-reencode-fails branch:
repair geometry/action ranking or use tokenized/local controls before more
planner scale or hierarchy.

Grid 5 oversight is re-enabled for the current Grid5 wave only.

- Prompt: `scripts/oversight/grid5_oversight_prompt.md`
- Wrapper: `scripts/slurm/run_grid5_oversight.slurm`
- Invocation: sources `~/.bash_profile`, then uses the local `cs` alias as
  `cs ... exec` with `model_reasoning_effort="medium"`
- Dummy verified job: `3724787`, exit `0:0`, no edits
- Scheduled jobs every 6h for 2.5 days. `3724789`, `3724790`, and `3724791`
  completed cleanly; later jobs are pending by begin time:
  `3724792` at 2026-06-13 16:50,
  `3724793` at 2026-06-13 22:50,
  `3724794` at 2026-06-14 04:50,
  `3724795` at 2026-06-14 10:50,
  `3724796` at 2026-06-14 16:50,
  `3724797` at 2026-06-14 22:50,
  `3724798` at 2026-06-15 04:50.

## Grid 5 Matrix

All variants train JEPA latent MSE plus SIGReg and a learned terminal-energy
head by default.

- Encoder: `mlp` vs `cls_transformer`
- Predictor: one-hidden-layer `mlp` vs causal `ar_transformer`
- Dynamics target: full next latent vs residual delta
- Latent size: `32`, `64`, `128`

Each task automatically runs diagnostics after training:

- latent distribution/SIGReg stats
- oracle latent-goal distance along oracle trajectories
- learned terminal-energy calibration along trajectories
- adjacent/all-action ranking under oracle latent distance and learned energy
- concrete JSONL action examples
- small enumerated beam planning under oracle latent distance and learned energy

Diagnostic artifacts are written under each run root in `diagnostics/`.

## Latest Grid 5 Read

The solve gate failed:

- oracle `latent_goal` beam planning: `0/16` solves for all variants
- learned `goal_energy` beam planning: `0/16` solves for all variants
- best oracle remaining Hamming:
  `grid5_sigreg_mlp_mlp_delta_z128`, mean `44.88`
- best learned-energy remaining Hamming:
  `grid5_sigreg_mlp_mlp_delta_z64`, mean `48.19`

The main diagnostic pattern is monotone gold trajectories but poor all-action
ranking. For the best oracle variant, latent and learned-energy monotone rates
are both `0.992`, but oracle latent top-1 gold action is only `0.031`, oracle
latent top action is goal-correct only `0.156`, learned-energy top-1 gold is
`0.000`, and learned-energy top action is goal-correct only `0.063`.

The completed Grid 5 diagnostics used a small enumerated beam, not LeWorldModel
MPC-CEM. The posthoc `3724325` job added the CEM/MPC lookahead control and also
failed: all 24 checkpoints solved `0` at every horizon. Average remaining
Hamming improved slightly with horizon, from about `53` at h4 to about `51.5`
at h64, but no run reached terminal boards or exact solves.

Recursive rollout training also failed the solve gate. Both delta
`3724413_[0-5]` and full-state `3724500_[0-5]` completed cleanly, wrote all
standard and MPC-CEM diagnostics, and solved `0` under every score/horizon.
Best MPC-CEM proximity was
`grid5_recursive_mlp_mlp_delta_z128_k2` with oracle `latent_goal` at h64, mean
remaining Hamming `49.88`. Best learned `goal_energy` proximity was
`grid5_recursive_mlp_ar_transformer_state_z128_k2` at h64, mean remaining
Hamming `50.50`. The recursive loss reduces the train/eval mismatch in
principle, but this sweep did not produce a planner-ready compact latent.

Latest local CPU probe:
`$PUZZLE_JEPA_WORK_ROOT/analysis/grid5_symbolic_probe_20260612/`,
`grid5_symbolic_probe_state_20260612/`, and
`grid5_symbolic_probe_true_hamming_20260612/`.

It removes learned predictor rollout from planning by executing candidate
futures symbolically, re-encoding the exact boards, and scoring them. This also
failed: `0/4` solves at horizons `8/16/32/64/full` for oracle `latent_goal` and
learned `goal_energy`, with mean remaining Hamming around `45-51`. The AR
full-state recursive checkpoint has much lower K=32 latent drift than the base
MLP-delta checkpoint, but symbolic re-encode planning still fails, so the
current blocker is not only predictor drift.

Latest Grid5B 10M read: capacity/stabilization improved directional signals but
not exact solving. `canonical_ema_vicreg_k4` is the best symbolic oracle
variant so far, with cheap beam oracle mean remaining Hamming `29.56`, latent
top-goal-value rate `0.969`, symbolic re-encode h8 mean remaining Hamming
`41.00`, and solve `0/4`. True-Hamming symbolic CEM can reach mean remaining
Hamming `1.75` and solve `1/4` on several variants, so the flat symbolic
optimizer is not hopeless, but the latent and learned scores still do not rank
solutions sharply enough.
