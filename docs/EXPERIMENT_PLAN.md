# Experiment Plan

Source of truth: `../sequence-editing-report/BACKLOG.md` and
`../sequence-editing-report/CURRENT_EXPERIMENTS.md`.

Wave 14 is complete and its representation gate failed. No new training sweep
is active or authorized.

The next bounded work is evaluation repair on retained checkpoints:

1. Calibrate frozen property probes with standardized regression targets or a
   closed-form ridge baseline and verify convergence against raw and matched
   initialization controls.
2. Replace the joint 256D-state/8D-macro nearest-neighbor support score with a
   conditional macro-support diagnostic that can separate held-out valid from
   synthetic off-support chunks.
3. Only for a cell surviving those checks, rerun enough planning episodes to
   report uncertainty rather than three binary trials.

A later training ablation may compare staged encoder freezing with joint
high-level gradients. That is required before claiming hierarchy itself induces
abstract state features. Predictor, capacity, LDAD, SIGReg, object-load, and
trajectory grids remain blocked.
