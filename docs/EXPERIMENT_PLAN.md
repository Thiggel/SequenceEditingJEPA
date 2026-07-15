# Experiment Plan

Source of truth: `../sequence-editing-report/BACKLOG.md` and
`../sequence-editing-report/CURRENT_EXPERIMENTS.md`.

Wave 14 is complete, but staged encoder freezing made it the wrong protocol for
the intended hierarchy experiment. No new sweep is active or authorized.

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
Before submission, add SIGReg to the controlled model, test that every level
backpropagates into the shared encoder, repair regression-probe calibration,
and replace the state-dominated support score. Planning then runs only on
representation-qualified cells, on a fixed shared episode set with enough
episodes for confidence intervals. Predictor, capacity, object-load, and
trajectory grids remain blocked.
