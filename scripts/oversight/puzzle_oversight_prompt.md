You are running scheduled autonomous oversight for `/home/hpc/c107fa/c107fa12/sequence-editing`.

Read first:
- `AGENTS.md`
- `docs/RUNBOOK.md`
- `docs/RESULTS.md`
- `docs/EXPERIMENT_PLAN.md`
- `../sequence-editing-report/BACKLOG.md`
- `../sequence-editing-report/STATUS.md`
- `../sequence-editing-report/RESULTS.md`
- `../sequence-editing-report/LOG.md`
- `../sequence-editing-report/report.tex`
- `../sequence-editing-report/notes/legacy.md`

Use `source scripts/env.sh` for shell commands. Do not change package versions. Do not modify `../legacy-sequence-editing` except to inspect archived files if needed. Keep changes surgical, test them, and commit/push both repos after successful verification.

Core loop for every oversight run:
- Reconcile live Slurm state with the experiment plan and backlog. Use `squeue`, recent `sacct`, `scontrol show job`, logs, run roots, metrics, checkpoints, diagnostics, and output artifacts. Restrict status to jobs from this repo unless another job directly affects these experiments.
- Update `docs/RUNBOOK.md`, `docs/RESULTS.md`, `docs/EXPERIMENT_PLAN.md`, `../sequence-editing-report/BACKLOG.md`, `../sequence-editing-report/STATUS.md`, `../sequence-editing-report/RESULTS.md`, `../sequence-editing-report/LOG.md`, and `../sequence-editing-report/report.tex` whenever jobs finish, fail, get submitted, or new interpretations are found.
- Question assumptions explicitly. Check whether a metric is actually measuring solve quality, whether online metrics disagree with diagnostics, whether oracle information is leaking, whether planning uses latent rollout or re-encoded symbolic states, whether invalid states/actions are being handled as intended, and whether current gates still follow from the data.
- Inspect concrete examples, not only aggregate metrics. For puzzle JEPA this means planner traces, terminal boards, goal boards, mismatch records, remaining Hamming, latent-vs-reencoded planning records, and any sample-generation outputs available in run roots. If diagnostics lack examples needed to understand a failure, add the smallest diagnostic extension and test it.
- Analyze as much as possible from finished jobs. Produce concise but meaningful tables and plots under `../sequence-editing-report/assets/` when they clarify a decision: drift curves, goal-rank/action-rank summaries, terminal remaining-Hamming distributions, mismatch heatmaps, latent-vs-reencoded planning comparisons, training curves, and failure tables. Reference useful figures/tables in `report.tex`.
- If a job failed, timed out, OOMed, or produced incomplete artifacts, inspect the traceback and resource usage. If the fix is small and local, implement it, run focused pytest/smoke checks, update docs, and resubmit only the failed job or a safer replacement. For OOM, lower the most relevant batch size first.
- If jobs completed and the backlog gate is satisfied, submit the next documented experiment. If the gate is not satisfied, write why and add a concrete next diagnostic or ablation to `BACKLOG.md`.
- During housekeeping of pending jobs, check whether other suitable GPU partitions appear freer. If useful and the job is not dependency- or begin-time-blocked, try `scontrol update JobId=<jobid> Partition=<partition1,partition2>`.
- Ensure the oversight chain continues every 4 hours. The Slurm wrapper should submit the next run automatically; still verify that exactly one later `puzzle_oversight` job is pending or scheduled. If none exists, submit `scripts/slurm/puzzle_oversight.slurm` with `--begin=now+4hours`. If duplicate stale oversight jobs exist, cancel only stale superseded oversight jobs and record it.

Current active focus:
- Check `puzzle_grid3b`, `puzzle_diag3b_large`, `puzzle_diag3b_n2`, and `puzzle_oversight` jobs first.
- Grid 3B large diagnostics for `sudoku_jepa_5m_local_direct_weighted` completed as `3680019`: latent rollout planning solved `0/64`, re-encoded symbolic-state planning solved `64/64`, and terminal-only scoring only changed latent terminal fill rate from `1/64` to `4/64`. Treat this as evidence that the lead checkpoint's remaining oracle-goal failure is latent rollout drift / stale latent state, not the local action scorer.
- Reuse `../sequence-editing-report/assets/grid3b/` when interpreting the lead diagnostics: planning comparison, drift curve, terminal mismatch heatmap, CSV summaries, and concrete latent failure examples are already there.
- If Grid 3B rollout `N=2` finishes, ensure dependent diagnostics `3680021` ran or are submitted. When diagnostics finish, compare against Grid 3A `local_direct_weighted` and Grid 3B lead large diagnostics on `goal_rank`, 10/20/terminal drift, latent-vs-reencoded planning, terminal solve, remaining Hamming, mismatch concentration, and training curves.
- Do not start Maze, 10M/20M size sweeps, or broad controls until the Grid 3B rollout `N=2` diagnostics finish, unless the evidence clearly shows the current Sudoku branch is blocked.

Historical checks to preserve:
- Grid 0 was infrastructure only.
- Grid 1/2A established rollout drift and weak action grounding as bottlenecks.
- Grid 3A showed local value-only action injection fixes sampled goal-action grounding, while exact terminal solve remains `0.0`.
- Residual prediction and changed-cell-only loss are rejected for the immediate next branch unless new evidence overturns that interpretation.

End with a compact status report: job IDs, states, output roots, checkpoints, latest metrics or failure reasons, artifacts/figures updated, docs updated, commits pushed, and next safe steps.
