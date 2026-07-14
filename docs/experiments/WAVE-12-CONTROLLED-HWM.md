# Wave 12: Controlled Transformer/CLS HWM

## Question

Can a controlled rigid-object world isolate hierarchy, rollout, capacity, and
Delta/LDAD planning behavior?

## Runs

- V1 jobs `3849807`-`3849879`, 72 completed jobs.
- Fidelity v2 jobs `3850221`-`3850274`.
- Identifiable v3 jobs `3850409`-`3850444`, 36 jobs.
- Replacement hierarchy/rollout jobs `3850619`-`3850672` were later canceled
  when the scope changed.
- Capacity extension covered token/CLS `64/32`, `128/64`, `256/128`.

## Results

V1 had invalid unchanged transitions and planner leaks; it is not evidence
against HWM. Corrected v2/v3 had positive prediction gains but no group passed
the action-ranking or 95% planning gates. Wider CLS improved position/relation
readout, while hierarchy still failed. Full-grid LDAD decoded actions but did
not improve forward control.

## Conclusion

Inverse action decoding and forward planning are separable. The wave still
confounded object load, encoder width, partial edit states, and hierarchy.
