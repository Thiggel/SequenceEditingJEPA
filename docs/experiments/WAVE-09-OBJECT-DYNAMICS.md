# Wave 09: Object Dynamics Scaffold

## Question

Do construction order, hierarchy, LDAD, EMA, VICReg, or SIGReg cause a
single-CLS JEPA to discover object/process variables?

## Runs

Commit `14de87a` introduced the object-dynamics scaffold. The final bounded
evaluation covered 315 train/dynamics/probe jobs plus corrected probes,
prestage stability replications, dual train/random-edit probes, hierarchy, and
the seven construction/completion/repair trajectories. Job history and the
135-job dual-probe gate are preserved in the external report.

## Results

EMA improved rollout transfer only. Reconstruction matched or beat static
semantic probes, temporal ordering was not causal, random-edit transfer failed,
and HWM CEM exact success was zero. Interleaved/global-random ordering often
beat coherent object-blocked ordering on aggregate probes.

## Conclusion

The scaffold learned scene/process summaries, not reliable object state. The
broad phase was retired in favor of moving-object and rate-controlled tests.
