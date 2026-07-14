# Wave 10: Moving-Object Bottlenecks

## Question

Does reducing latent capacity or increasing object load force moving-object
representations to encode shape, position, velocity, and relations?

## Runs

- 90 deterministic reflected runs across latent width and object load.
- Temporal-delta gate jobs `3834849`-`3834872`, diagnostics
  `3834947`-`3834970`.
- Transfer trainers `3836223`-`3836276` and diagnostics/probes.
- Reconstruction controls `3837715`-`3838120`.
- Exact-load reflected matrix: 90 runs, 30 groups, three seeds.

## Results

No exact-load JEPA group passed all-seed shape, position, velocity, or relation
gates. Exact N8 lost shape at every width. A z4/N8 predictor beat identity but
was nonsemantic. Reconstruction showed that z32 could carry layout/position,
so decoder capacity was not the primary limitation.

## Conclusion

Small bottlenecks do not automatically bind objects. Predictable low-rank
statistics can win the JEPA loss without carrying the requested factors.
