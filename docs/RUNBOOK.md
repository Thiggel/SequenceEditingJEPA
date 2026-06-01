# Runbook

Last updated: 2026-06-01 15:10 CEST

Long-form handoff source of truth: `../sequence-editing-report`.

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
| `3688542` | PENDING | Exactly one later enhanced recurring oversight, begin-time-blocked until `2026-06-01 16:57:08 CEST`. |
| `3688587_[0-2]` | CANCELLED | User-approved cancellation at `2026-06-01 14:46:56 CEST`; pre-HWM-correction Grid 4A baseline jobs ran `01:40:56` and wrote step-1 metrics. |
| `3688921_[0-2]` | CANCELLED | Superseded after the user requested exact report-style hierarchical planning; cancelled at `2026-06-01 15:01:20 CEST` after `00:14:07` on `a0531`, `a0731`, and `a0931`; no checkpoints. |
| `3688986_[0-2]` | RUNNING | Exact-recipe Grid 4A training resubmitted at `2026-06-01 15:09:33 CEST`; `_0` on `a0531`, `_1` and `_2` on `a0731`; output roots are `sudoku_jepa_5m_goal_energy_hwm_l1`, `..._l2_span9`, `..._l3_span3`. |

Check live state:

```bash
squeue -j 3688542,3688587,3688921,3688986 -o "%.18i %.9T %.28j %.10M %.20S %R"
sacct -j 3688587,3688921,3688986 --format=JobID,JobName%30,State,ExitCode,Elapsed,Start,End,NodeList
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
oversight `3687722` started at `2026-06-01 12:57:00 CEST`, and exactly one
later oversight, `3688542`, is pending for `2026-06-01 16:57:08 CEST`.
Other visible HFSA/paired user-account arrays are outside this repo snapshot.
Partition housekeeping at 15:09 CEST: `sinfo` showed idle `a40` nodes and idle
`a100` capacity, but the Grid 4A jobs require `--constraint=a100_80` and started
on `a100` immediately. `3688542` is begin-time-blocked, so no
`scontrol update ... Partition=...` is useful.

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
Use `scripts/slurm/run_grid4a_subgoal_cem_diagnostics.slurm` after l2/l3
checkpoints exist. The older `run_grid4a_hierarchical_cem_diagnostics.slurm`
is only a comparison diagnostic that scores primitive candidate chunks through
the action encoder; it is not the exact report recipe.

Focused validation passed under the repo venv:

```bash
source scripts/env.sh
pytest tests/test_puzzle_models.py tests/test_puzzle_hydra.py -q
```

The 13:06 CEST oversight rerun of that pytest command was interrupted before
collection because Python imports were stuck in shared filesystem waits
(`rpc_wait_bit_killable`/`folio_wait_bit_common`). The Slurm wrapper syntax
checks and `py_compile` for the changed Grid 4A modules passed.

After Grid 4A checkpoints exist, run:

```bash
sbatch scripts/slurm/run_grid4a_cem_diagnostics.slurm
```

For the report-style oracle-goal hierarchical subgoal diagnostic on corrected
l2/l3 checkpoints, run:

```bash
sbatch scripts/slurm/run_grid4a_subgoal_cem_diagnostics.slurm
```

For the older primitive-candidate hierarchy comparison diagnostic, run
`scripts/slurm/run_grid4a_hierarchical_cem_diagnostics.slurm` only after the
exact subgoal planner is recorded or if a direct comparison is needed.

Oversight uses `scripts/oversight/puzzle_oversight_prompt.md`. That prompt
requires each run to reconcile Slurm/artifacts with the backlog, inspect
concrete planner examples, question assumptions, add useful report figures and
tables, fix/resubmit small failures, and keep the four-hour oversight chain
alive. The next safe experiment is the Grid 4A goal-energy/hierarchy/CEM grid;
keep the oracle-goal reset result as the control baseline and do not start Maze
or broad capacity sweeps yet.
