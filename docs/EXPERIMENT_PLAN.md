# Experiment Plan

Source of truth: `../sequence-editing-report/BACKLOG.md` and
`../sequence-editing-report/CURRENT_EXPERIMENTS.md`.

The active plan has three independent gates:

1. Calibrate no-VISReg LDAD, VICReg, and SIGReg recipes over multipliers
   `{1,10,100}` and online/stop-gradient/EMA target contracts.
2. Test whole trajectories and dense joint hierarchy supervision at
   `T={100,300,500}`, including a constant-processed-state batch axis.
3. Diagnose retained VICReg/EMA+VICReg planners with conditional support
   projection and lower-level reachability feedback over 32 shared episodes.

Do not combine winners or open broader predictor, capacity, object-load, or
trajectory-type axes until these three manifests are complete and compared
over all seeds. Never add a full-grid latent row.
