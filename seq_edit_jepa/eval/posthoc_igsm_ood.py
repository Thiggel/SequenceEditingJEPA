from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

from seq_edit_jepa.data.corruptors import build_corruptor
from seq_edit_jepa.data.tasks import build_task
from seq_edit_jepa.eval.sample_printing import format_generation_samples
from seq_edit_jepa.models import CausalTransformerLM, DenoisingLM, SequenceEditJEPA
from seq_edit_jepa.eval.full_denoise import evaluate_full_denoise
from seq_edit_jepa.train.config import load_yaml
from seq_edit_jepa.train.evaluate import evaluate
from seq_edit_jepa.train.lm_evaluate import evaluate_causal_lm


DEFAULT_RUNS = [
    "igsm_official_med_mask_action_conditioned_jepa_200k",
    "igsm_official_med_mask_denoising_lm_200k",
    "igsm_official_med_causal_lm_200k",
]


def main() -> None:
    args = _parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    root = Path(args.runs_root or os.environ.get("SEQ_EDIT_JEPA_WORK_ROOT", "outputs")).expanduser()
    if root.name != "runs" and (root / "runs").exists():
        root = root / "runs"
    run_dirs = [root / name for name in (args.runs or DEFAULT_RUNS)]
    output = Path(args.output or _default_output_path()).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    with open(output, "w", encoding="utf-8") as handle:
        for run_dir in run_dirs:
            checkpoints = _checkpoint_dirs(run_dir, args.checkpoint_glob)
            if args.latest_only:
                checkpoints = checkpoints[-1:]
            for checkpoint in checkpoints:
                for split_name, split in _eval_splits(args):
                    metrics = evaluate_checkpoint_on_split(
                        checkpoint,
                        split=split,
                        split_name=split_name,
                        seq_len=args.seq_len,
                        batches=args.batches,
                        batch_size=args.batch_size,
                        device=device,
                        ood_op_values=args.ood_op_values,
                        modulus=args.modulus,
                        full_denoise=args.full_denoise,
                        commit_k=args.commit_k,
                        jepa_inference_mode=args.jepa_inference_mode,
                        print_samples=args.print_samples,
                        sample_examples=args.sample_examples,
                        sample_max_chars=args.sample_max_chars,
                    )
                    row = {
                        "run": run_dir.name,
                        "checkpoint": checkpoint.name,
                        "step": _checkpoint_step(checkpoint),
                        "split_name": split_name,
                        "split": split,
                        "commit_k": args.commit_k,
                        "jepa_inference_mode": args.jepa_inference_mode,
                        **metrics,
                    }
                    rows.append(row)
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    handle.flush()
                    print(json.dumps(row, sort_keys=True), flush=True)

    summary = _summarize(rows)
    summary_path = output.with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps({"metrics": str(output), "summary": str(summary_path)}, sort_keys=True))


def evaluate_checkpoint_on_split(
    checkpoint: Path,
    split: str,
    split_name: str,
    seq_len: int,
    batches: int,
    batch_size: int,
    device: torch.device,
    ood_op_values: list[int],
    modulus: int | None,
    full_denoise: bool,
    commit_k: str,
    jepa_inference_mode: str,
    print_samples: bool,
    sample_examples: int,
    sample_max_chars: int,
) -> dict[str, float]:
    run_dir = checkpoint.parent
    config = load_yaml(run_dir / "config.yaml")
    task_cfg = dict(config.get("task", {}))
    task_cfg["ood_op_values"] = list(ood_op_values)
    task_cfg["ood_modulus"] = modulus
    if split in {"eval_ood", "test_ood"} or split.startswith("op_") or split.startswith("eval_op_"):
        task_cfg["modulus"] = modulus
    tokenizer_path = run_dir / "tokenizer"
    if (tokenizer_path / "vocab.json").exists():
        task_cfg["_tokenizer_path"] = str(tokenizer_path)
    task = build_task(task_cfg)
    tokenizer = task.tokenizer

    model = _load_model(checkpoint, device)
    _allow_rope_length_extrapolation(model, seq_len)
    eval_cfg = {
        "split": split,
        "batches": batches,
        "batch_size": batch_size,
        "rollout_steps": [],
        "commit_k": commit_k,
        "jepa_inference_mode": jepa_inference_mode,
    }
    architectures = set(getattr(model.config, "architectures", []) or [])
    if "CausalTransformerLM" in architectures or isinstance(model, CausalTransformerLM):
        metrics = evaluate_causal_lm(model, task, tokenizer, seq_len, eval_cfg, device)
    elif full_denoise:
        corruptor = build_corruptor(dict(config.get("corruptor", {})), tokenizer)
        metrics = evaluate_full_denoise(model, task, tokenizer, corruptor, seq_len, eval_cfg, device, prefix="eval/full_denoise")
    else:
        corruptor = build_corruptor(dict(config.get("corruptor", {})), tokenizer)
        metrics = evaluate(model, task, corruptor, seq_len, eval_cfg, device)
    if print_samples:
        sample_corruptor = build_corruptor(dict(config.get("corruptor", {"name": "mask", "num_steps": 16})), tokenizer) if not isinstance(model, CausalTransformerLM) else None
        print(
            format_generation_samples(
                model,
                task,
                tokenizer,
                seq_len,
                device,
                corruptor=sample_corruptor,
                splits=[split],
                examples_per_split=sample_examples,
                max_chars=sample_max_chars,
                commit_k=commit_k,
                jepa_inference_mode=jepa_inference_mode,
            ),
            flush=True,
        )
    return {f"{split_name}/{key.removeprefix('eval/')}": float(value) for key, value in metrics.items()}


def _load_model(checkpoint: Path, device: torch.device):
    raw_config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    architectures = set(raw_config.get("architectures", []))
    if "CausalTransformerLM" in architectures:
        cls = CausalTransformerLM
    elif "DenoisingLM" in architectures:
        cls = DenoisingLM
    else:
        cls = SequenceEditJEPA
    return cls.from_pretrained(checkpoint).to(device)


def _allow_rope_length_extrapolation(model, seq_len: int) -> None:
    if hasattr(model, "config"):
        model.config.max_position_embeddings = max(int(getattr(model.config, "max_position_embeddings", seq_len)), int(seq_len))
    for module_name in ("encoder", "target_encoder", "predictor"):
        module = getattr(model, module_name, None)
        if module is not None and hasattr(module, "max_length"):
            module.max_length = max(int(getattr(module, "max_length")), int(seq_len))


def _checkpoint_dirs(run_dir: Path, glob_pattern: str) -> list[Path]:
    checkpoints = sorted((path for path in run_dir.glob(glob_pattern) if path.is_dir()), key=_checkpoint_step)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints matching {glob_pattern!r} under {run_dir}")
    return checkpoints


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return -1


def _eval_splits(args: argparse.Namespace) -> list[tuple[str, str]]:
    splits = [("ood_ops", "eval_ood")]
    if args.by_op:
        splits.extend((f"op_{op}", f"eval_op_{op}") for op in args.ood_op_values)
    if args.include_id:
        splits.insert(0, ("id", "eval"))
    return splits


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for row in rows:
        key = f"{row['run']}/{row['split_name']}"
        summary[key] = {
            metric: value
            for metric, value in row.items()
            if isinstance(value, (int, float)) and metric not in {"step"}
        }
    return summary


def _default_output_path() -> str:
    root = Path(os.environ.get("SEQ_EDIT_JEPA_WORK_ROOT", "outputs")) / "posthoc" / "igsm_ood"
    root.mkdir(parents=True, exist_ok=True)
    return str(root / "metrics.jsonl")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved iGSM checkpoints on op-count OOD splits.")
    parser.add_argument("--runs-root", default=None)
    parser.add_argument("--runs", nargs="*", default=None)
    parser.add_argument("--checkpoint-glob", default="checkpoint-*")
    parser.add_argument("--latest-only", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--ood-op-values", nargs="+", type=int, default=[20, 21, 22, 23])
    parser.add_argument("--modulus", type=int, default=23)
    parser.add_argument("--by-op", action="store_true")
    parser.add_argument("--include-id", action="store_true")
    parser.add_argument("--full-denoise", action="store_true")
    parser.add_argument("--commit-k", default="schedule", help="Use scheduled commits or a fixed number of tokens per denoising step.")
    parser.add_argument(
        "--jepa-inference-mode",
        choices=["predictor_decoder", "policy_head"],
        default="predictor_decoder",
        help="How to turn JEPA hidden states into token predictions during full denoising.",
    )
    parser.add_argument("--print-samples", action="store_true")
    parser.add_argument("--sample-examples", type=int, default=1)
    parser.add_argument("--sample-max-chars", type=int, default=1600)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    main()
