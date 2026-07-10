from __future__ import annotations

import json
from pathlib import Path

import pytest

from puzzle_jepa.eval.object_dynamics_results import summarize_object_dynamics_runs, write_summary


def test_object_dynamics_summary_uses_fixed_step_zero_deltas(tmp_path: Path) -> None:
    run = tmp_path / "run_a"
    run.mkdir()
    _write_config(run, seed=7, max_steps=10)
    _write_metrics(
        run,
        [
            _record(0, std=0.5, current=0.4, object_map=0.1),
            _record(5, std=0.75, current=0.5, object_map=0.2, loss=0.3),
            _record(10, std=1.0, current=0.6, object_map=0.25, loss=0.1),
        ],
    )
    (run / "checkpoint.pt").touch()

    summary = summarize_object_dynamics_runs(tmp_path)

    assert summary["run_count"] == 1
    assert summary["complete_run_count"] == 1
    endpoint = summary["checkpoints"][-1]
    assert endpoint["complete"]
    assert endpoint["latent_std_ratio"] == pytest.approx(2.0)
    assert endpoint["delta_probe_current_object_acc"] == pytest.approx(0.2)
    assert endpoint["delta_probe_object_map_foreground_miou"] == pytest.approx(0.15)


def test_object_dynamics_summary_aggregates_seeds_and_writes_outputs(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    for seed, current in ((7, 0.5), (11, 0.7)):
        run = root / f"run_{seed}"
        run.mkdir(parents=True)
        _write_config(run, seed=seed, max_steps=10)
        _write_metrics(run, [_record(0, std=0.5, current=0.4, object_map=0.1), _record(10, std=0.5, current=current, object_map=0.2, loss=0.2)])
        (run / "checkpoint.pt").touch()
        _write_balanced_reprobe(run, current_delta=current - 0.4)

    summary = summarize_object_dynamics_runs(root)
    aggregate = summary["endpoint_aggregates"][0]
    assert aggregate["probe_fit_version"] == 1
    assert aggregate["seeds"] == [7, 11]
    assert aggregate["complete_n"] == 2
    assert aggregate["delta_probe_current_object_acc_mean"] == pytest.approx(0.2)
    assert aggregate["delta_probe_current_object_acc_std"] == pytest.approx(0.1)
    balanced = summary["balanced_reprobe_aggregates"][0]
    assert balanced["probe_fit_version"] == 3
    assert balanced["seeds"] == [7, 11]
    assert balanced["delta_probe_current_object_acc_mean"] == pytest.approx(0.2)

    output = tmp_path / "summary"
    write_summary(summary, output)
    assert json.loads((output / "object_dynamics_summary.json").read_text())["run_count"] == 2
    assert "dCurrent" in (output / "object_dynamics_summary.md").read_text()


def test_object_dynamics_summary_rejects_mislabeled_probe_schema(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    _write_config(run, seed=7, max_steps=10)
    _write_metrics(run, [_record(0, std=0.5, current=0.4, object_map=0.1)])
    (run / "checkpoint.pt").touch()
    _write_balanced_reprobe(run, current_delta=0.1)
    result = json.loads((run / "probe_eval_balanced_v3.json").read_text())
    (run / "probe_eval_balanced_v2.json").write_text(json.dumps(result))
    (run / "probe_eval_balanced_v3.json").unlink()

    summary = summarize_object_dynamics_runs(tmp_path)

    assert summary["balanced_reprobe_count"] == 0


def _write_config(run: Path, *, seed: int, max_steps: int) -> None:
    config = {
        "seed": seed,
        "data": {"name": "T0_semantic_mix"},
        "model": {"name": "M0_cls64_r1"},
        "objective": {"name": "base"},
        "training": {"learning_rate": 1.0e-4, "max_steps": max_steps},
    }
    (run / "config.json").write_text(json.dumps(config))


def _write_metrics(run: Path, records: list[dict[str, float | int]]) -> None:
    (run / "metrics.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))


def _write_balanced_reprobe(run: Path, *, current_delta: float) -> None:
    result = {
        "probe_fit_version": 3,
        "checkpoint_step": 10,
        "initial_latent_std_mean": 0.5,
        "latent_std_mean": 0.5,
        "initial_probe_current_object_acc": 0.4,
        "delta_probe_current_object_acc": current_delta,
        "delta_probe_object_count_acc": 0.1,
        "delta_probe_object_count_balanced_acc": 0.1,
        "delta_probe_current_object_balanced_acc": current_delta,
        "delta_probe_delta_action_object_acc": 0.1,
        "delta_probe_object_map_foreground_miou": 0.1,
        "delta_probe_grid_foreground_miou": 0.1,
        "delta_rollout_error_invalid_auroc": 0.1,
    }
    (run / "probe_eval_balanced_v3.json").write_text(json.dumps(result))


def _record(
    step: int,
    *,
    std: float,
    current: float,
    object_map: float,
    loss: float | None = None,
) -> dict[str, float | int]:
    record: dict[str, float | int] = {
        "step": step,
        "latent_std_mean": std,
        "latent_effective_rank": 4.0,
        "probe_object_count_acc": 0.5,
        "probe_current_object_acc": current,
        "probe_delta_action_object_acc": 0.5,
        "probe_object_map_foreground_miou": object_map,
        "probe_grid_foreground_miou": 0.2,
        "rollout_error_invalid_auroc": 0.6,
    }
    if loss is not None:
        record["train_loss"] = loss
    return record
