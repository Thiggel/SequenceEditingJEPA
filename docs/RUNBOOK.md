# Runbook

Last updated: 2026-05-28

## Environment

```bash
source scripts/env.sh
```

This activates `$WORK/.venv`, sets local cache directories, and adds the repo to
`PYTHONPATH`. Runtime outputs default to `/home/vault/$(id -gn)/$USER/sequence-editing`
when writable, otherwise `$WORK/sequence-editing`.

## Current Repo Shape

The legacy sequence-editing/iGSM code was moved out of the active tree and
archived at:

```text
../legacy-sequence-editing
```

The legacy results are summarized in [`../legacy.md`](../legacy.md). Active code
is only the puzzle-world scaffold under `puzzle_jepa`.

## Validation

Run the full active test suite:

```bash
python -m pytest -q tests
```

Run smoke experiments:

```bash
python -m puzzle_jepa.train.hydra_train --config-name jepa_sudoku_smoke
python -m puzzle_jepa.train.hydra_train --config-name jepa_maze_smoke
python -m puzzle_jepa.train.hydra_train --config-name hrm_sudoku_smoke
python -m puzzle_jepa.train.hydra_train --config-name trm_sudoku_smoke
python -m puzzle_jepa.train.hydra_train --config-name ptrm_sudoku_smoke
```

## Active Modules

| Module | Purpose |
| --- | --- |
| `puzzle_jepa.data.worlds` | Sudoku/Maze objective state spaces, legal actions, state transitions, goal checks. |
| `puzzle_jepa.data.trajectories` | Oracle one-step transition sampling and tensor collation. |
| `puzzle_jepa.data.hf_puzzles` | Hugging Face CSV/string adapters for flattened Sudoku/Maze examples. |
| `puzzle_jepa.models.action_jepa` | Decoder-free action-conditioned JEPA world model. |
| `puzzle_jepa.models.recursive` | Minimal HRM, TRM, and PTRM scaffolds. |
| `puzzle_jepa.planning.latent_planner` | Symbolic action enumeration and latent action ranking. |
| `puzzle_jepa.eval.diagnostics` | Grid 1 checkpoint diagnostics for action rank, latent unroll drift, terminal energy, and planner traces. |

## Slurm Status

## Handoff Snapshot: 2026-05-28 09:55 CEST

Grid 0 and Grid 1 completed successfully. All puzzle training tasks wrote final
metrics and checkpoints; there were no OOMs or Slurm failures. Grid 1 is still
an infrastructure/data-curriculum result, not a solver result: every final
H=1/2/4 solve-rate metric is `0.0`. Sudoku planning now uses explicit clue masks:
original puzzle cells are immutable, while non-clue cells may be overwritten and
may temporarily violate Sudoku constraints. Preliminary diagnostics `3666870`
completed with the old fill-only Sudoku action space; they showed strongly
non-monotonic predicted goal energy under oracle latent unrolls. Corrected
diagnostics `3667044_[0-4]` completed with clue-mask mutable Sudoku actions,
bounded terminal LeWM-style beam planning, and latent-energy plots. The concrete
bottleneck is now clearer: true re-encoded oracle states move monotonically to
the goal, but predicted latent rollouts drift far from the goal by terminal
steps, and bounded beam planning reaches no terminal boards/paths.
Oversight `3669988` completed cleanly after submitting recurring oversight
`3670421`, pending for `2026-05-28 12:12:52 CEST`. There are no active Grid 0,
Grid 1, or diagnostics jobs. A compile-ready LaTeX diagnostic report now lives
at `docs/puzzle_jepa_diagnostics_report.tex`; copied figure assets are under
`docs/assets/puzzle-diagnostics/`.

| Job ID | Name | State | Configs | Output roots | Notes |
| --- | --- | --- | --- | --- | --- |
| `3664581_0` | `puzzle_grid0` | COMPLETED, exit `0:0`, `2026-05-26 16:18:59-16:23:58 CEST` on `a0833` | `grid0_sudoku_jepa_5m_oracle_smoke` | `$PUZZLE_JEPA_WORK_ROOT/runs/grid0_sudoku_jepa_5m_oracle_smoke` | 5.26M trainable params. Final step `1000`: train loss `0.0142`, eval loss `0.0141`, oracle-action top1 `0.03125`, mean rank `30.56`, H=1 solve rate `0.0`; `checkpoint.pt` and `checkpoint-1000.pt` exist. |
| `3664581_1` | `puzzle_grid0` | COMPLETED, exit `0:0`, `2026-05-26 16:18:59-16:23:58 CEST` on `a0833` | `grid0_maze_jepa_5m_oracle_smoke` | `$PUZZLE_JEPA_WORK_ROOT/runs/grid0_maze_jepa_5m_oracle_smoke` | 5.28M trainable params. Final step `1000`: train loss `0.0168`, eval loss `0.0163`, oracle-action top1 `0.0`, mean rank `116.5`, H=1 solve rate `0.0`; `checkpoint.pt` and `checkpoint-1000.pt` exist. |
| `3665018_[0-4]` | `puzzle_grid1` | COMPLETED, exit `0:0`, `2026-05-26 20:29:34-21:33:41 CEST` on `a0932` | `grid1_sudoku_jepa_5m_oracle`, `grid1_sudoku_jepa_5m_mix70_30`, `grid1_sudoku_jepa_5m_mix50_50`, `grid1_maze_jepa_5m_oracle`, `grid1_maze_jepa_5m_mix70_30` | `$PUZZLE_JEPA_WORK_ROOT/runs/{sudoku_jepa_5m_oracle,sudoku_jepa_5m_mix70_30,sudoku_jepa_5m_mix50_50,maze_jepa_5m_oracle,maze_jepa_5m_mix70_30}` | All five final `metrics.json`, `metrics.jsonl`, `checkpoint.pt`, and `checkpoint-5000.pt` files exist. Stderr files are empty. Sudoku max GPU memory was `9572 MiB`; Maze max GPU memory was `26676 MiB` oracle and `22834 MiB` mix70/30. |
| `3664583` | `puzzle_oversight` | COMPLETED, exit `0:0`, `2026-05-26 20:19:09-20:34:31 CEST` on `a0932` | `scripts/slurm/puzzle_oversight.slurm` | `logs/puzzle_oversight_3664583.{out,err}` | Submitted oversight `3665011` before invoking `cs exec`. |
| `3665011` | `puzzle_oversight` | COMPLETED, exit `0:0`, `2026-05-27 00:19:19-00:24:06 CEST` on `a0731` | `scripts/slurm/puzzle_oversight.slurm` | `logs/puzzle_oversight_3665011.{out,err}` | Submitted recurring oversight `3665528`; no code changes were needed. |
| `3665528` | `puzzle_oversight` | COMPLETED, exit `0:0`, `2026-05-27 04:19:47-04:24:36 CEST` on `a0833` | `scripts/slurm/puzzle_oversight.slurm` | `logs/puzzle_oversight_3665528.{out,err}` | Submitted recurring oversight `3665918`; no code changes were needed. |
| `3665918` | `puzzle_oversight` | COMPLETED, exit `0:0`, `2026-05-27 08:20:32-08:24:55 CEST` on `a0632` | `scripts/slurm/puzzle_oversight.slurm` | `logs/puzzle_oversight_3665918.{out,err}` | Submitted recurring oversight `3666371`; no code changes were needed. |
| `3666371` | `puzzle_oversight` | CANCELLED at `2026-05-27 12:10:47 CEST` | old copy of `scripts/slurm/puzzle_oversight.slurm` | n/a | Replaced because the submitted Slurm copy did not include the new `puzzle_diag` oversight instructions. |
| `3666870_[0-4]` | `puzzle_diag` | COMPLETED, exit `0:0`, `2026-05-27 12:10:51-12:38:34 CEST` | first diagnostics version | `$PUZZLE_JEPA_WORK_ROOT/runs/{sudoku_jepa_5m_oracle,sudoku_jepa_5m_mix70_30,sudoku_jepa_5m_mix50_50,maze_jepa_5m_oracle,maze_jepa_5m_mix70_30}/diagnostics` | Preliminary diagnostics with old fill-only Sudoku planning. Useful drift signal: predicted goal energy was non-monotonic under oracle latent unrolls (`0.29-0.44` monotone rate), while true re-encoded oracle states were almost perfectly monotonic. Sudoku mutable action-rank/planner traces from this job are superseded by `3667044`. |
| `3667030_[0-4]` | `puzzle_diag` | CANCELLED at `2026-05-27 13:06:07 CEST` | intermediate corrected diagnostics | n/a | Replaced before completion to add actual terminal LeWM-style planning diagnostics. |
| `3667038_[0-4]` | `puzzle_diag` | CANCELLED at `2026-05-27 13:09:28 CEST` | intermediate corrected diagnostics | n/a | Replaced before completion so the final diagnostics also include latent-energy PNG plots. |
| `3667044_[0-4]` | `puzzle_diag` | COMPLETED, exit `0:0`, `2026-05-27 13:09:38-13:48:08 CEST` on `a0633`, `a0731`, `a0831`, `a0832` | `scripts/slurm/run_grid1_diagnostics.slurm` | same diagnostics roots as above | Final corrected diagnostics. All five `diagnostics/diagnostics.json`, `rank_records.jsonl`, `drift_records.jsonl`, and `latent_energy_mse.png` outputs exist; planner traces are embedded in `diagnostics.json` rather than written as standalone JSONL. Stderr files are empty. Sudoku mutable rank is weak (`top1` `0.0088-0.0127`, mean rank `167.68-184.79`); Maze rank is also weak (`top1` `0.0078-0.0195`, mean rank `124.12-205.27`). Predicted latent energy is non-monotonic (`0.287-0.436` monotone rate), true re-encoded oracle energy is monotonic (`0.999-1.000`), and both step-energy and terminal-energy beam planning have `0.0` solve/terminal rate. |
| `3666871` | `puzzle_oversight` | COMPLETED, exit `0:0`, `2026-05-27 16:10:49-16:20:09 CEST` on `a0731` | updated `scripts/slurm/puzzle_oversight.slurm` | `logs/puzzle_oversight_3666871.{out,err}` | Submitted recurring oversight `3667453`; no code changes were needed. |
| `3667453` | `puzzle_oversight` | COMPLETED, exit `0:0`, `2026-05-27 20:11:30-20:16:39 CEST` on `a0731` | updated `scripts/slurm/puzzle_oversight.slurm` | `logs/puzzle_oversight_3667453.{out,err}` | Submitted recurring oversight `3668489`; no code changes were needed. |
| `3668489` | `puzzle_oversight` | COMPLETED, exit `0:0`, `2026-05-28 00:11:53-00:15:48 CEST` on `a0731` | updated `scripts/slurm/puzzle_oversight.slurm` | `logs/puzzle_oversight_3668489.{out,err}` | Submitted recurring oversight `3669194`; no code changes were needed. |
| `3669194` | `puzzle_oversight` | COMPLETED, exit `0:0`, `2026-05-28 04:12:26-04:16:54 CEST` on `a0731` | updated `scripts/slurm/puzzle_oversight.slurm` | `logs/puzzle_oversight_3669194.{out,err}` | Submitted recurring oversight `3669988`; no code changes were needed. |
| `3669988` | `puzzle_oversight` | COMPLETED, exit `0:0`, `2026-05-28 08:12:46-08:17:04 CEST` on `a0731` | updated `scripts/slurm/puzzle_oversight.slurm` | `logs/puzzle_oversight_3669988.{out,err}` | Submitted recurring oversight `3670421`; no code changes were needed. |
| `3670421` | `puzzle_oversight` | PENDING, begin time `2026-05-28 12:12:52 CEST` | updated `scripts/slurm/puzzle_oversight.slurm` | `logs/puzzle_oversight_3670421.{out,err}` | Next recurring oversight. |

Grid 1 final metrics:

| Run | Eval loss | Top1 | Mean rank | H1/H2/H4 solve | Oracle delta | Random delta | Checkpoint |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| `sudoku_jepa_5m_oracle` | `0.0085` | `0.0625` | `36.50` | `0.0`/`0.0`/`0.0` | `0.0084` | `-0.0008` | `checkpoint.pt`, `checkpoint-5000.pt` |
| `sudoku_jepa_5m_mix70_30` | `0.0084` | `0.03125` | `28.16` | `0.0`/`0.0`/`0.0` | `0.0087` | `-0.0022` | `checkpoint.pt`, `checkpoint-5000.pt` |
| `sudoku_jepa_5m_mix50_50` | `0.0074` | `0.03125` | `49.03` | `0.0`/`0.0`/`0.0` | `0.0079` | `-0.0001` | `checkpoint.pt`, `checkpoint-5000.pt` |
| `maze_jepa_5m_oracle` | `0.0021` | `0.0` | `16.00` | `0.0`/`0.0`/`0.0` | `0.0014` | `-0.0003` | `checkpoint.pt`, `checkpoint-5000.pt` |
| `maze_jepa_5m_mix70_30` | `0.0019` | `0.0` | `245.50` | `0.0`/`0.0`/`0.0` | `0.0009` | `-0.0005` | `checkpoint.pt`, `checkpoint-5000.pt` |

Corrected Grid 1 diagnostics:

| Run | Rank top1/top5 | Mean rank | Predicted energy monotone | True energy monotone | Oracle-unroll terminal predicted MSE | Step/terminal beam solve |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `sudoku_jepa_5m_oracle` | `0.0088`/`0.0303` | `184.79` | `0.334` | `1.000` | `1.753` | `0.0`/`0.0` |
| `sudoku_jepa_5m_mix70_30` | `0.0127`/`0.0420` | `182.01` | `0.352` | `0.999` | `1.729` | `0.0`/`0.0` |
| `sudoku_jepa_5m_mix50_50` | `0.0127`/`0.0361` | `167.68` | `0.287` | `1.000` | `1.710` | `0.0`/`0.0` |
| `maze_jepa_5m_oracle` | `0.0078`/`0.0781` | `124.12` | `0.372` | `1.000` | `1.938` | `0.0`/`0.0` |
| `maze_jepa_5m_mix70_30` | `0.0195`/`0.0391` | `205.27` | `0.436` | `1.000` | `1.956` | `0.0`/`0.0` |

Interpretation: Grid 0 and Grid 1 pass infrastructure criteria. One-step losses
fall sharply, and oracle transitions are consistently closer to the goal latent
than their starting states, but raw oracle-goal latent distance still does not
produce final-step solves. The diagnostics indicate that latent rollout drift
and weak early oracle-action ranking are the immediate bottlenecks. Do not move
to Grid 2 size ablations until a re-encode/rollout diagnostic or scorer change
creates a measurable planning signal, or until a concrete planner/evaluation bug
is found and fixed.

Validation before Grid 1 submission:

```bash
python -m pytest -q tests
python -m compileall -q puzzle_jepa
python -m puzzle_jepa.train.grid0 --config-name grid1_sudoku_jepa_5m_mix70_30 ... training.max_steps=1 ...
python -m puzzle_jepa.train.grid0 --config-name grid1_maze_jepa_5m_mix70_30 ... training.max_steps=1 ...
```

Commands:

```bash
sacct -j 3664581,3665018,3664583,3665011,3665528,3665918,3666371,3666870,3667030,3667038,3667044,3666871,3667453,3668489,3669194,3669988,3670421 --format=JobID,JobName%30,State,ExitCode,Elapsed,Start,End,NodeList
squeue -j 3670421 -o "%.18i %.9P %.30j %.8T %.20S %.10M %.6D %R"
cat "$PUZZLE_JEPA_WORK_ROOT/runs/grid0_sudoku_jepa_5m_oracle_smoke/metrics.json"
cat "$PUZZLE_JEPA_WORK_ROOT/runs/grid0_maze_jepa_5m_oracle_smoke/metrics.json"
cat "$PUZZLE_JEPA_WORK_ROOT/runs/sudoku_jepa_5m_oracle/metrics.json"
cat "$PUZZLE_JEPA_WORK_ROOT/runs/sudoku_jepa_5m_mix70_30/metrics.json"
cat "$PUZZLE_JEPA_WORK_ROOT/runs/sudoku_jepa_5m_mix50_50/metrics.json"
cat "$PUZZLE_JEPA_WORK_ROOT/runs/maze_jepa_5m_oracle/metrics.json"
cat "$PUZZLE_JEPA_WORK_ROOT/runs/maze_jepa_5m_mix70_30/metrics.json"
find "$PUZZLE_JEPA_WORK_ROOT/runs" -path "*/diagnostics/diagnostics.json" -print
find "$PUZZLE_JEPA_WORK_ROOT/runs" -path "*/diagnostics/latent_energy_mse.png" -print
```

The old `seqedit` jobs and oversight jobs are complete; see `legacy.md`.
