# Wave 02: Grid4 Energy and Hierarchy

## Question

Can learned goal energies, scalar values, hierarchical subgoals, macro actions,
or CEM replace exact symbolic scoring in Sudoku planning?

## Runs

Commits `4fa4a68` through `779df4f` added Grid4A-Grid4T families: goal-energy
hierarchy CEM, HWM-style action hierarchy, exact hierarchical subgoal CEM,
learned-energy reset beam, oracle reset calibration, contrastive goal energy,
candidate-rank diagnostics, CVL/ListNet/reachability value objectives, scorer
spread, macro bottlenecks, global single-latent MLPs, and mixed rollouts.
Exact job-by-job IDs and configs are preserved chronologically in the external
report `LOG.md`.

## Results

Exact symbolic and true-Hamming scoring could solve. Learned goal distance,
learned scalar energy/value, CEM, and the tested macro hierarchies did not
recover robust exact planning. Oracle reset controls localized the failure to
learned latent geometry and rollout, rather than the transition simulator.

## Conclusion

Search sophistication did not repair a goal geometry that failed to rank
successors. Later waves therefore separated oracle geometry, predicted goals,
action grounding, and latent rollout.
