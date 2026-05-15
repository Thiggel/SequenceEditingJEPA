# Sequence-Edit JEPA

This branch implements a minimal action-conditioned latent world model for sequence editing:

```text
whole sequence state -> edit action -> latent transition -> revised sequence
```

The first fully runnable experiments are fixed-length demasking and replacement correction on controlled LANO/CFG-style strings and synthetic iGSM-style arithmetic traces. Variable-length insert/delete editing is implemented at the symbolic action layer and exposed for later model extensions.

Use the shared cluster environment:

```bash
source ./scripts/env.sh
python -m pytest
python -m seq_edit_jepa.train.hydra_train --config-name smoke_lano_mask
```

Training uses Hydra config names, a Hugging Face `Trainer` subclass, and HF-compatible `PreTrainedModel` checkpoints. Each run saves:

```text
$SEQ_EDIT_JEPA_WORK_ROOT/runs/<experiment>/
  config.yaml
  metrics.json
  checkpoint.pt          # compatibility checkpoint
  model/                 # save_pretrained directory
  tokenizer/
```

The default architecture is a bidirectional sequence-edit Transformer with RoPE, RMSNorm, QK-norm self-attention, SwiGLU MLPs, sinusoidal timestep conditioning, an edit-policy MLP head, and an action-conditioned latent predictor. Debug sequence prints are controlled by the `debug:` config block.

SLURM launchers live under `scripts/slurm/` and use the same `$WORK/.venv`, proxy, and cache settings copied from `../T-JEPA`. Runtime outputs go under `SEQ_EDIT_JEPA_WORK_ROOT`, defaulting to `$WORK/sequence-editing`.

Current experiment tracking:

```text
docs/EXPERIMENT_PLAN.md
docs/RESULTS.md
docs/DENOISING_LITERATURE.md
docs/RUNBOOK.md
docs/sequence_edit_jepa_report.tex
```
