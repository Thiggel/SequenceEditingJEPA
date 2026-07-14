# Wave 01: Grid3 Local JEPA

## Question

Can a local action-conditioned JEPA learn Sudoku edit dynamics and support
planning when the goal is encoded by the same model?

## Runs

- Primary archived run: `sudoku_jepa_5m_local_direct_weighted_rollout_n2`.
- `5,000` optimizer steps, peak LR `1e-4`, weight decay `.05`.
- Local-value action conditioning: row/column select one cell and the digit
  embedding is added at that cell.
- One-step transition batch `768` plus rollout batch `512` in the N=2 run.
- Local/context loss weights were changed cell `8`, row/column/block `2`, all
  other cells `1`.

## Results

Oracle-goal symbolic re-encoding solved `64/64` and `128/128`. Resetting the
latent every four actions also solved both sets. Uninterrupted latent rollout
solved only `4/64` and `7/128` in the two archived diagnostics.

## Conclusion

The historical “100%” result was a symbolic-reset planning result, not a
stable latent world-model rollout. It remains a useful action-grounding
control, but not evidence for long-horizon JEPA planning.

Source: report `RESULTS.md`, section `Historical Local-Value Audit`.
