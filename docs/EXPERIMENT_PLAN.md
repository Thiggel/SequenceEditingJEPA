# Experiment Plan

Source of truth: `../sequence-editing-report/BACKLOG.md` and
`../sequence-editing-report/CURRENT_EXPERIMENTS.md`.

Wave 15 is active. It corrects Wave 14's staged encoder freeze by training all
three hierarchy levels and the shared encoder jointly from step 0.

The corrected bounded experiment keeps data, model, `[1,10,100]` spans,
rollout supervision, bottleneck, and capacity fixed. Every level, action
encoder, predictor, and the shared state encoder trains jointly from step 0.
At three seeds, compare:

1. bare online JEPA and EMA-only controls;
2. SIGReg and EMA+SIGReg;
3. VICReg and EMA+VICReg using the selected Wave 14 coefficient pair;
4. paper-style online LDAD, EMA+LDAD, and LDAD combined separately with
   VICReg or SIGReg, with and without EMA.

This is a 12-objective, 36-run training gate, not another coefficient grid.
SIGReg, per-level encoder-gradient tests, standardized regression probes, and
conditional macro support are implemented. Trainers are job `3858542`; probes
are `3858543`. Planning runs only on representation-qualified cells, on a fixed
shared episode set with confidence intervals. Predictor, capacity, object-load,
and trajectory grids remain blocked.
