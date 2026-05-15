from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import TrainingArguments

from seq_edit_jepa.data.corruptors import build_corruptor
from seq_edit_jepa.data.tasks import build_task
from seq_edit_jepa.eval.full_denoise import evaluate_full_denoise_with_splits
from seq_edit_jepa.eval.sample_printing import format_generation_samples
from seq_edit_jepa.models import SequenceEditJEPA, build_model
from seq_edit_jepa.train.config import default_output_dir, load_yaml, save_yaml
from seq_edit_jepa.train.debug import DebugSequenceCallback
from seq_edit_jepa.train.evaluate import evaluate
from seq_edit_jepa.train.hf_data import CorruptionDataCollator, CorruptionIterableDataset
from seq_edit_jepa.train.hf_trainer import EMAUpdateCallback, SeqEditTrainer
from seq_edit_jepa.train.periodic_eval import PeriodicSampleCallback, PeriodicTaskEvalCallback
from seq_edit_jepa.train.tokenizer_io import load_or_build_tokenizer


def run_experiment(config_path: str | Path) -> dict[str, float]:
    config = load_yaml(config_path)
    return run_experiment_from_config(config, source_config_path=config_path)


def run_experiment_from_config(config: dict[str, Any], source_config_path: str | Path | None = None) -> dict[str, float]:
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
    corruptor = build_corruptor(dict(config.get("corruptor", {})), tokenizer)
    model = build_model(dict(config.get("model", {})), tokenizer, num_steps=corruptor.num_steps, max_length=seq_len)

    training = dict(config.get("training", {}))
    eval_cfg = dict(config.get("eval", {}))
    train_dataset = CorruptionIterableDataset(task, corruptor, seq_len, "train", length=None)
    eval_examples = int(eval_cfg.get("dataset_examples", int(eval_cfg.get("batch_size", 32)) * int(eval_cfg.get("batches", 4))))
    eval_dataset = CorruptionIterableDataset(task, corruptor, seq_len, "eval", length=eval_examples)
    args = _training_args(training, output_dir)
    callbacks = [EMAUpdateCallback()]
    debug_cfg = dict(config.get("debug", {}))
    if bool(debug_cfg.get("enabled", False)):
        callbacks.append(
            DebugSequenceCallback(
                task=task,
                corruptor=corruptor,
                tokenizer=tokenizer,
                seq_len=seq_len,
                every_steps=int(debug_cfg.get("every_steps", training.get("log_every", 100))),
                rollout_steps=int(debug_cfg.get("rollout_steps", 0)),
                max_actions=int(debug_cfg.get("max_actions", 24)),
            )
        )
    periodic_every = int(eval_cfg.get("every_steps", 0))
    if periodic_every > 0:
        periodic_eval_cfg = dict(eval_cfg)
        periodic_eval_cfg["batches"] = int(eval_cfg.get("periodic_batches", eval_cfg.get("batches", 4)))
        periodic_eval_cfg["extra_splits"] = _periodic_extra_splits(eval_cfg, periodic_eval_cfg["batches"])
        callbacks.append(
            PeriodicTaskEvalCallback(
                evaluator=lambda current_model, current_device: _evaluate_seqedit(
                    current_model,
                    task,
                    tokenizer,
                    corruptor,
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
                    corruptor=corruptor,
                    splits=sample_cfg.get("splits", ["eval"]),
                    examples_per_split=int(sample_cfg.get("examples", 1)),
                    trace_steps=sample_cfg.get("trace_steps", [corruptor.num_steps, max(1, corruptor.num_steps // 2), 1]),
                    max_chars=int(sample_cfg.get("max_chars", 1600)),
                    step=current_step,
                    commit_k=sample_cfg.get("commit_k", eval_cfg.get("commit_k", None)),
                    jepa_inference_mode=str(sample_cfg.get("jepa_inference_mode", eval_cfg.get("jepa_inference_mode", "predictor_decoder"))),
                ),
                every_steps=int(sample_cfg.get("every_steps", periodic_every or training.get("log_every", 100))),
            )
        )

    trainer = SeqEditTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CorruptionDataCollator(),
        callbacks=callbacks,
        task=task,
        corruptor=corruptor,
        seq_len=seq_len,
        rollout_steps=int(training.get("rollout_steps", config.get("model", {}).get("rollout_steps", 0))),
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    eval_device = args.device if isinstance(args.device, torch.device) else torch.device(args.device)
    metrics = _evaluate_seqedit(model, task, tokenizer, corruptor, seq_len, eval_cfg, eval_device)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    trainer.save_model(str(output_dir / "model"))
    tokenizer.save_pretrained(output_dir / "tokenizer")
    tokenizer.save_pretrained(output_dir / "model")
    torch.save({"model": model.state_dict(), "config": config}, output_dir / "checkpoint.pt")
    hub_cfg = dict(config.get("hub", {}))
    if bool(hub_cfg.get("push_to_hub", False)):
        repo_id = str(hub_cfg["repo_id"])
        private = bool(hub_cfg.get("private", False))
        model.push_to_hub(repo_id, private=private)
        if hasattr(tokenizer, "push_to_hub"):
            tokenizer.push_to_hub(repo_id, private=private)
    return metrics


def _evaluate_seqedit(model, task, tokenizer, corruptor, seq_len: int, eval_cfg: dict[str, Any], device: torch.device) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if bool(eval_cfg.get("one_step", True)):
        metrics.update(evaluate(model, task, corruptor, seq_len, eval_cfg, device))
    if bool(eval_cfg.get("full_denoise", False)):
        metrics.update(evaluate_full_denoise_with_splits(model, task, tokenizer, corruptor, seq_len, eval_cfg, device))
    return metrics


def _periodic_extra_splits(eval_cfg: dict[str, Any], default_batches: int) -> list[dict[str, Any]]:
    splits = []
    for split in eval_cfg.get("extra_splits", []):
        child = dict(split)
        child["batches"] = int(child.get("periodic_batches", child.get("batches", default_batches)))
        splits.append(child)
    return splits


def _training_args(training: dict[str, Any], output_dir: Path) -> TrainingArguments:
    max_steps = int(training.get("max_steps", 1000))
    save_steps = int(training.get("save_steps", max(1, max_steps)))
    use_cuda = torch.cuda.is_available()
    return TrainingArguments(
        output_dir=str(output_dir),
        report_to=training.get("report_to", "none"),
        remove_unused_columns=False,
        max_steps=max_steps,
        per_device_train_batch_size=int(training.get("batch_size", 32)),
        per_device_eval_batch_size=int(training.get("eval_batch_size", training.get("batch_size", 32))),
        learning_rate=float(training.get("lr", 3e-4)),
        weight_decay=float(training.get("weight_decay", 0.01)),
        max_grad_norm=float(training.get("grad_clip", 1.0)),
        logging_steps=max(1, int(training.get("log_every", 50))),
        save_steps=save_steps,
        save_total_limit=int(training.get("save_total_limit", 2)),
        bf16=bool(training.get("bf16", False)) and use_cuda,
        fp16=bool(training.get("fp16", False)) and use_cuda,
        dataloader_num_workers=int(training.get("dataloader_num_workers", 0)),
        dataloader_pin_memory=bool(training.get("dataloader_pin_memory", False)),
        ignore_data_skip=bool(training.get("ignore_data_skip", True)),
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
