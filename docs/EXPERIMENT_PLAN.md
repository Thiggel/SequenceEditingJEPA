# Experiment Plan

Source of truth: `../sequence-editing-report/BACKLOG.md` and
`../sequence-editing-report/CURRENT_EXPERIMENTS.md`.

The plan has three independent gates:

1. Calibrate no-VISReg LDAD, VICReg, and SIGReg recipes over multipliers
   `{1,10,100}` and online/stop-gradient/EMA target contracts.
2. Test whole trajectories and dense joint hierarchy supervision at
   `T={100,300,500}`, including a constant-processed-state batch axis.
3. Diagnose retained VICReg/EMA+VICReg planners with conditional support
   projection and lower-level reachability feedback over 32 shared episodes.

The objective and planner gates are complete and did not identify a component
to promote: objective strengths expose a semantics-versus-rank tradeoff, and
planner constraints hurt baseline CEM. Finish the repaired dense manifest
before deciding whether whole-trajectory supervision changes that conclusion.
Do not open broader predictor, capacity, object-load, or trajectory-type axes
yet. Never add a full-grid latent row.
