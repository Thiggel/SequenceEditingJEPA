# Wave 11: Hard Rate and Trajectory Transfer

## Question

Under a literal quantized rate ceiling, which factors survive, and are they
stable across trajectory types?

## Runs

- Exact-load trajectory matrix: 168 rows; trainers `3841078`-`3841245`,
  dynamics `3841266`-`3841433`, probes `3841434`-`3841602`.
- Corrected hard-rate matrix: 108 runs across z2/z4/z8, N2/N4/N8, and quantizer
  levels 0/2/4/16; trainers `3844346`-`3844453`, dynamics `3844454`-`3844561`,
  probes `3844562`-`3844669`.
- Final transfer: 162 runs, 54 groups, three seeds across wrapped, rotating,
  and seven construction/completion/repair families.

## Results

At exact N8, z4/q16 and z8/q16 learned weak shape; z8/q2 and z8/q4 learned
weak color-indexed position. Matched nominal bit rates selected different
factors. Reconstruction recovered position but not shape. Transfer changed or
removed these factors; no row jointly retained shape, position, dynamics,
completion, and relations.

## Conclusion

This is evidence for trajectory-sensitive compact factors, not a general
object representation. Unique colors still provide object-slot shortcuts.
