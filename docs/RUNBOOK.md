# Runbook

Last updated: 2026-06-12 14:00 CEST

Long-form handoff source of truth: `../sequence-editing-report`.

- Ongoing LaTeX report: `../sequence-editing-report/report.tex`
- Experiment backlog: `../sequence-editing-report/BACKLOG.md`
- Live status: `../sequence-editing-report/STATUS.md`
- Results and insights: `../sequence-editing-report/RESULTS.md`
- Chronological log: `../sequence-editing-report/LOG.md`

## Active Surface

The active experiment surface has been reset to Grid 5.

- Config: `configs/puzzle/grid5_sudoku_sigreg.yaml`
- Slurm: `scripts/slurm/run_grid5_sigreg_ablation.slurm`
- Model: `puzzle_jepa/models/sigreg_jepa.py`
- Train: `puzzle_jepa/train/grid5.py`
- Diagnostics: `puzzle_jepa/eval/grid5_diagnostics.py`

Old `grid0`-`grid4` experiment configs and Slurm wrappers were removed from the
active tree. Historical results remain in `../sequence-editing-report`.

## Environment

```bash
source scripts/env.sh
pytest tests/test_grid5_sigreg.py tests/test_puzzle_hydra.py -q
```

Runtime outputs default to:

```text
/home/vault/$(id -gn)/$USER/sequence-editing
```

## Slurm Snapshot

Grid 5 was submitted as `3722613_[0-23]` at 2026-06-12 11:29 CEST and has
completed.

- Partition request: `a40,a100,rtxpro6k`
- Resource request: one GPU per task, 8 CPUs, 8h wall time
- Final state: all 24 tasks completed with exit code `0:0`
- Runtime: about 10-12 minutes on `rtxpro6k`, 20-27 minutes on `a40`
- Stderr: all Grid 5 stderr files are empty
- Output roots:
  `$PUZZLE_JEPA_WORK_ROOT/runs/grid5_sigreg_{encoder}_{predictor}_{state|delta}_z{32|64|128}`

Legacy Grid 4Z `3722524` is still running but is now superseded by Grid 5. Do
not extend Grid 4 unless explicitly requested. Grid 4Q `3715252_[0-11]` remains
pending with `DependencyNeverSatisfied`; it is not consuming resources.

Grid 5 posthoc MPC-CEM lookahead diagnostics were submitted as
`3724325_[0-23]` at 2026-06-12 13:44 CEST.

- Wrapper: `scripts/slurm/run_grid5_mpc_cem_diagnostics.slurm`
- Eval module: `puzzle_jepa/eval/grid5_mpc_cem_diagnostics.py`
- Initial state: tasks `_0`-`_19` running on `rtxpro6k`; `_20`-`_23` pending
- Purpose: LeWorldModel-style MPC-CEM over horizons `4/8/16/32/64`
- Outputs:
  `$PUZZLE_JEPA_WORK_ROOT/runs/grid5_sigreg_*/diagnostics_mpc_cem/`

Grid 5 recursive rollout training was submitted as `3724413_[0-5]` at
2026-06-12 14:00 CEST.

- Wrapper: `scripts/slurm/run_grid5_recursive_rollout.slurm`
- Initial state: pending across `a40,a100,rtxpro6k`
- Fixed base: MLP encoder, delta prediction, latent size `128`
- Factors: predictor `mlp|ar_transformer` x recursive rollout K `2|4|8`
- Outputs:
  `$PUZZLE_JEPA_WORK_ROOT/runs/grid5_recursive_mlp_{predictor}_delta_z128_k{K}`
- Each task trains, runs standard Grid 5 diagnostics, then runs MPC-CEM
  horizons `4/8/16/32/64` with oracle `latent_goal` and learned `goal_energy`.

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
MPC-CEM. The posthoc `3724325` job is the CEM/MPC lookahead control.
