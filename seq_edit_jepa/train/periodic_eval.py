from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import torch
from transformers import TrainerCallback


class PeriodicTaskEvalCallback(TrainerCallback):
    """Run task-specific evaluation during training, not only at the end."""

    def __init__(
        self,
        evaluator: Callable[[torch.nn.Module, torch.device], dict[str, float]],
        every_steps: int = 0,
        filename: str = "periodic_eval_metrics.jsonl",
    ):
        self.evaluator = evaluator
        self.every_steps = int(every_steps)
        self.filename = filename

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.every_steps <= 0 or model is None:
            return control
        if state.global_step <= 0 or state.global_step % self.every_steps != 0:
            return control
        if not getattr(state, "is_local_process_zero", True) and int(getattr(args, "local_process_index", 0)) not in {-1, 0}:
            return control
        device = next(model.parameters()).device
        metrics = self.evaluator(model, device)
        metrics = {f"periodic/{key.removeprefix('eval/')}": float(value) for key, value in metrics.items()}
        metrics["periodic/step"] = int(state.global_step)
        path = Path(args.output_dir) / self.filename
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, sort_keys=True) + "\n")
        print(json.dumps(metrics, sort_keys=True), flush=True)
        return control


class PeriodicSampleCallback(TrainerCallback):
    """Print qualitative generation samples during training."""

    def __init__(
        self,
        sampler: Callable[[torch.nn.Module, torch.device, int], str],
        every_steps: int = 0,
        filename: str = "periodic_eval_samples.txt",
    ):
        self.sampler = sampler
        self.every_steps = int(every_steps)
        self.filename = filename

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.every_steps <= 0 or model is None:
            return control
        if state.global_step <= 0 or state.global_step % self.every_steps != 0:
            return control
        if not getattr(state, "is_local_process_zero", True) and int(getattr(args, "local_process_index", 0)) not in {-1, 0}:
            return control
        device = next(model.parameters()).device
        text = self.sampler(model, device, int(state.global_step))
        path = Path(args.output_dir) / self.filename
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(text.rstrip() + "\n")
        print(text, flush=True)
        return control
