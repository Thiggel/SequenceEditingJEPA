# Agent Handoff Rules

- The long-form handoff source of truth lives outside this repo at
  `../sequence-editing-report`.
- Always update `../sequence-editing-report/report.tex`,
  `../sequence-editing-report/BACKLOG.md`,
  `../sequence-editing-report/STATUS.md`,
  `../sequence-editing-report/RESULTS.md`, and/or
  `../sequence-editing-report/LOG.md` whenever experiments, implementation,
  Slurm state, results, or interpretation change.
- Keep this repo's `docs/RUNBOOK.md`, `docs/RESULTS.md`, and
  `docs/EXPERIMENT_PLAN.md` concise. They should point to the report repo and
  hold only the latest operational snapshot needed by an agent starting in this
  repo.
- Keep a dedicated current-sweep summary in
  `../sequence-editing-report/CURRENT_EXPERIMENTS.md`, with a compact mirror in
  `docs/CURRENT_EXPERIMENTS.md`. When the user asks "how is it going?", answer
  from this current-sweep summary: first summarize what this sweep is testing
  and what it means, then give result tables, then give analysis/insights.
- For Slurm work, include job IDs, config/run names, state, output roots,
  checkpoints, latest meaningful metrics, and failure reasons in the report repo
  and in the compact in-repo snapshot.
- Maintain `../sequence-editing-report/BACKLOG.md` as the experiment backlog.
  Add new experiments when they are proposed, update their status when submitted
  or completed, and record the gate/decision that determines the next run.
- Maintain `../sequence-editing-report/LOG.md` as a short chronological log of
  major events only: new experiment grids submitted, cancellations, important
  fixes, failures, final results, and interpretation changes. Do not log routine
  conversational Q&A or every status request.
- Only aggregate results into `RESULTS.md`, `STATUS.md`, and
  `CURRENT_EXPERIMENTS.md` when new data, new insights, major Slurm state
  changes, or new experiment grids exist. Avoid churn from logging every
  interaction.
- During housekeeping of pending Slurm jobs, check whether other suitable GPU
  partitions appear freer. When appropriate, try broadening pending jobs with
  `scontrol update JobId=<jobid> Partition=<partition1,partition2>` to reduce
  wait time. Do not do this for dependency- or begin-time-blocked jobs where it
  cannot help.
- After successful verification, commit and push changes in both this repo and
  `../sequence-editing-report` to GitHub. If pushing fails, report the exact
  reason and leave the commits locally.
- Do not delete active Slurm job logs or cancel active jobs unless explicitly
  asked. It is OK to cancel stale superseded oversight jobs and delete ignored
  local clutter/logs after preserving useful results in the report repo.
- Delta-JEPA experiment planning invariant: every Delta-JEPA variant must be
  represented by both a full-grid latent run and a single learned-CLS latent
  run. Autonomous oversight follow-ups that propose or tune Delta-JEPA must
  add the paired single-latent variant and keep train/eval scripts in sync.

# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
