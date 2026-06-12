# Runbook

Last updated: 2026-06-12 11:30 CEST

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

## Active Slurm Snapshot

Grid 5 was submitted as `3722613_[0-23]` at 2026-06-12 11:29 CEST.

- Partition request: `a40,a100,rtxpro6k`
- Resource request: one GPU per task, 8 CPUs, 8h wall time
- Initial state: tasks `_0`-`_19` running on `rtxpro6k`; tasks `_20`-`_23`
  pending for priority
- Output roots:
  `$PUZZLE_JEPA_WORK_ROOT/runs/grid5_sigreg_{encoder}_{predictor}_{state|delta}_z{32|64|128}`

Legacy Grid 4Z `3722524` is still running but is now superseded by Grid 5. Do
not extend Grid 4 unless explicitly requested. Grid 4Q `3715252_[0-11]` remains
pending with `DependencyNeverSatisfied`; it is not consuming resources.

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
