# Wave 08: ARC JEPA Scaffold

## Question

Can ARC candidate transformations be trained and ranked as latent transitions?

## Runs

CPU coverage scaffold plus three first-pass train jobs `3821438`-`3821440`.
The active-context rerun retained candidate energy supervision while later code
masked direct target-positive records from transition dynamics.

## Results

First pass@1 was `0%`, `0%`, and `6.25%`. Audit found that target-positive
records with `action=None` contaminated the historical transition objective.

## Conclusion

The first result is invalid for JEPA dynamics. Reruns are gated on same-episode
listwise ranking and improved candidate generation; no further ARC scale-up was
submitted.
