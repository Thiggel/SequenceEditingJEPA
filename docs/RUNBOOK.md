# Runbook

Long-form handoff source of truth: `../sequence-editing-report`.

Last updated: 2026-07-15 09:58 CEST

No controlled-object experiment is active. Wave 14 jobs `3855790`-`3855793`
completed all 192 tasks with exit `0:0`.

Task manifest:
`$HPCVAULT/sequence-editing/runs/controlled_objects/manifests/controlled_valid_hwm_vicreg_v1_steps20000_tasks.tsv`.
Output root:
`$HPCVAULT/sequence-editing/runs/controlled_objects/controlled_valid_hwm_vicreg_v1_steps20000/`.
The aggregate is `summary.json`; every final run has `probe_eval_v4.json`.

Forty-eight final `[1,10,100]` checkpoints are retained for evaluation repair.
The 96 redundant `[1]` and `[1,10]` checkpoints were removed after completion.
Do not launch a new training grid until the probe calibration and conditional
macro-support issues in `docs/BACKLOG.md` are resolved and the user selects the
next gate.
