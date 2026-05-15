from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import Trainer

from seq_edit_jepa.data.tasks import build_task
from seq_edit_jepa.eval.sample_printing import format_generation_samples
from seq_edit_jepa.models import build_model
from seq_edit_jepa.train.config import default_output_dir, load_yaml, save_yaml
from seq_edit_jepa.train.hf_experiment import _training_args
from seq_edit_jepa.train.lm_data import CleanLMDataCollator, CleanLMIterableDataset
from seq_edit_jepa.train.lm_evaluate import evaluate_causal_lm
from seq_edit_jepa.train.periodic_eval import PeriodicSampleCallback, PeriodicTaskEvalCallback
from seq_edit_jepa.train.tokenizer_io import load_or_build_tokenizer


def run_lm_experiment(config_path: str | Path) -> dict[str, float]:
    config = load_yaml(config_path)
    return run_lm_experiment_from_config(config, source_config_path=config_path)


def run_lm_experiment_from_config(config: dict[str, Any], source_config_path: str | Path | None = None) -> dict[str, float]:
    seed = int(config.get("experiment", {}).get("seed", 0))
    _set_seed(seed)
    output_dir = default_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(config, output_dir / "config.yaml")
    if source_config_path is not None:
        shutil.copyfile(source_config_path, output_dir / "source_config.yaml")
    task_config = dict(config.get("task", {}))
    resume_from_checkpoint = config.get("experiment", {}).get("resume_from_checkpoint")
    if resume_from_checkpoint and (output_dir / "tokenizer" / "vocab.json").exists():
        task_config["_tokenizer_path"] = str(output_dir / "tokenizer")
    task = build_task(task_config)
    tokenizer = load_or_build_tokenizer(task, task_config, output_dir, prefer_saved=bool(resume_from_checkpoint))
    tokenizer.save_pretrained(output_dir / "tokenizer")
    seq_len = int(config.get("task", {}).get("seq_len", config.get("model", {}).get("max_length", 128)))
    model = build_model(dict(config.get("model", {})), tokenizer, num_steps=1, max_length=seq_len)
    training = dict(config.get("training", {}))
    eval_cfg = dict(config.get("eval", {}))
    eval_examples = int(eval_cfg.get("dataset_examples", int(eval_cfg.get("batch_size", 32)) * int(eval_cfg.get("batches", 4))))
    callbacks = []
    periodic_every = int(eval_cfg.get("every_steps", 0))
    if periodic_every > 0:
        periodic_eval_cfg = dict(eval_cfg)
        periodic_eval_cfg["batches"] = int(eval_cfg.get("periodic_batches", eval_cfg.get("batches", 4)))
        periodic_eval_cfg["extra_splits"] = _periodic_extra_splits(eval_cfg, periodic_eval_cfg["batches"])
        callbacks.append(
            PeriodicTaskEvalCallback(
                evaluator=lambda current_model, current_device: evaluate_causal_lm(
                    current_model,
                    task,
                    tokenizer,
                    seq_len,
                    periodic_eval_cfg,
                    current_device,
                ),
                every_steps=periodic_every,
            )
        )
    sample_cfg = dict(eval_cfg.get("sample_generations", {}))
    if bool(sample_cfg.get("enabled", False)):
        callbacks.append(
            PeriodicSampleCallback(
                sampler=lambda current_model, current_device, current_step: format_generation_samples(
                    current_model,
                    task,
                    tokenizer,
                    seq_len,
                    current_device,
                    splits=sample_cfg.get("splits", ["eval"]),
                    examples_per_split=int(sample_cfg.get("examples", 1)),
                    trace_steps=sample_cfg.get("trace_steps", []),
                    max_chars=int(sample_cfg.get("max_chars", 1600)),
                    step=current_step,
                ),
                every_steps=int(sample_cfg.get("every_steps", periodic_every or training.get("log_every", 100))),
            )
        )
    trainer = Trainer(
        model=model,
        args=_training_args(training, output_dir),
        train_dataset=CleanLMIterableDataset(task, seq_len, "train"),
        eval_dataset=CleanLMIterableDataset(task, seq_len, "eval", length=eval_examples),
        data_collator=CleanLMDataCollator(),
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    device = trainer.args.device if isinstance(trainer.args.device, torch.device) else torch.device(trainer.args.device)
    metrics = evaluate_causal_lm(model, task, tokenizer, seq_len, eval_cfg, device)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    trainer.save_model(str(output_dir / "model"))
    tokenizer.save_pretrained(output_dir / "model")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    torch.save({"model": model.state_dict(), "config": config}, output_dir / "checkpoint.pt")
    return metrics


def _periodic_extra_splits(eval_cfg: dict[str, Any], default_batches: int) -> list[dict[str, Any]]:
    splits = []
    for split in eval_cfg.get("extra_splits", []):
        child = dict(split)
        child["batches"] = int(child.get("periodic_batches", child.get("batches", default_batches)))
        splits.append(child)
    return splits


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
