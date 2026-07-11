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
        run = root / f"campaign_seed{seed}"
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
    assert aggregate["delta_probe_attention_entropy_mean"] == pytest.approx(0.1)
    balanced = summary["balanced_reprobe_aggregates"][0]
    assert balanced["probe_fit_version"] == 4
    assert balanced["run_family"] == "campaign"
    assert balanced["seeds"] == [7, 11]
    assert balanced["delta_probe_current_object_acc_mean"] == pytest.approx(0.2)
    assert balanced["raw_probe_action_process_provenance_acc_mean"] == pytest.approx(0.3)
    assert balanced["raw_probe_action_process_provenance_balanced_acc_mean"] == pytest.approx(0.4)
    assert balanced["probe_hierarchy_retrieval_goal_success_mean"] == pytest.approx(0.25)

    output = tmp_path / "summary"
    write_summary(summary, output)
    assert json.loads((output / "object_dynamics_summary.json").read_text())["run_count"] == 2
    markdown = (output / "object_dynamics_summary.md").read_text()
    assert "dCurrent" in markdown
    assert "| Probe | Distribution | Family | Model" in markdown
    assert "| v4 | T0_semantic_mix | campaign | M0_cls64_r1 | base | 1.0e-04 | 10 |" in markdown


def test_object_dynamics_summary_rejects_mislabeled_probe_schema(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    _write_config(run, seed=7, max_steps=10)
    _write_metrics(run, [_record(0, std=0.5, current=0.4, object_map=0.1)])
    (run / "checkpoint.pt").touch()
    _write_balanced_reprobe(run, current_delta=0.1)
    result = json.loads((run / "probe_eval_balanced_v4.json").read_text())
    (run / "probe_eval_balanced_v2.json").write_text(json.dumps(result))
    (run / "probe_eval_balanced_v4.json").unlink()

    summary = summarize_object_dynamics_runs(tmp_path)

    assert summary["balanced_reprobe_count"] == 0


def test_summary_does_not_pool_distinct_run_families(tmp_path: Path) -> None:
    for family, seed in (("calibration", 7), ("replication", 11)):
        run = tmp_path / f"{family}_seed{seed}"
        run.mkdir()
        _write_config(run, seed=seed, max_steps=10)
        _write_metrics(run, [_record(10, std=0.6, current=0.5, object_map=0.2, loss=0.1)])
        (run / "checkpoint.pt").touch()
        _write_balanced_reprobe(run, current_delta=0.2)

    summary = summarize_object_dynamics_runs(tmp_path)

    aggregates = summary["balanced_reprobe_aggregates"]
    assert [row["run_family"] for row in aggregates] == ["calibration", "replication"]
    assert [row["n"] for row in aggregates] == [1, 1]


def test_summary_keeps_common_and_in_domain_probe_distributions(tmp_path: Path) -> None:
    run = tmp_path / "campaign_seed7"
    run.mkdir()
    _write_config(run, seed=7, max_steps=10)
    _write_metrics(run, [_record(10, std=0.6, current=0.5, object_map=0.2, loss=0.1)])
    (run / "checkpoint.pt").touch()
    _write_balanced_reprobe(run, current_delta=0.2)
    in_domain = json.loads((run / "probe_eval_balanced_v4.json").read_text())
    in_domain["probe_trajectory_kind"] = "object_blocked"
    (run / "probe_eval_in_domain_v4.json").write_text(json.dumps(in_domain))

    summary = summarize_object_dynamics_runs(tmp_path)

    assert summary["balanced_reprobe_count"] == 2
    assert {row["probe_trajectory_kind"] for row in summary["balanced_reprobes"]} == {
        "T0_semantic_mix",
        "object_blocked",
    }
    assert len(summary["balanced_reprobe_aggregates"]) == 2
    output = tmp_path / "summary"
    write_summary(summary, output)
    markdown = (output / "object_dynamics_summary.md").read_text()
    assert "| v4 | T0_semantic_mix | campaign |" in markdown
    assert "| v4 | object_blocked | campaign |" in markdown


def test_summary_keeps_dependent_probe_when_inline_baseline_is_disabled(tmp_path: Path) -> None:
    run = tmp_path / "calibration"
    run.mkdir()
    _write_config(run, seed=7, max_steps=10)
    _write_metrics(run, [_record(10, std=0.6, current=0.5, object_map=0.2, loss=0.1)])
    (run / "checkpoint.pt").touch()
    _write_balanced_reprobe(run, current_delta=0.2)

    summary = summarize_object_dynamics_runs(tmp_path)

    assert summary["run_count"] == 1
    assert summary["complete_run_count"] == 1
    assert summary["balanced_reprobe_count"] == 1
    assert summary["balanced_reprobes"][0]["run_name"] == "calibration"


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
        "probe_fit_version": 4,
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
        "probe_delta_action_process_acc": 0.4,
        "raw_probe_action_process_provenance_acc": 0.3,
        "raw_probe_action_process_provenance_balanced_acc": 0.4,
        "probe_action_process_provenance_majority_acc": 0.5,
        "probe_action_process_provenance_majority_balanced_acc": 0.2,
        "probe_delta_action_process_balanced_acc": 0.35,
        "latent_nn_current_shape_acc": 0.2,
        "pixel_nn_current_shape_acc": 0.1,
        "latent_nn_current_color_acc": 0.4,
        "pixel_nn_current_color_acc": 0.3,
        "latent_nn_current_completion_mae": 0.2,
        "pixel_nn_current_completion_mae": 0.3,
        "probe_hierarchy_endpoint_mse": 0.01,
        "probe_hierarchy_level_agreement": 0.2,
        "probe_hierarchy_macro_retrieval_acc": 0.3,
        "probe_hierarchy_low_level_retrieval_acc": 0.25,
        "probe_hierarchy_retrieval_goal_success": 0.25,
        "probe_hierarchy_retrieval_goal_hamming": 0.05,
        "probe_hierarchy_cem_executed_goal_success": 0.0,
        "probe_hierarchy_cem_executed_goal_hamming": 0.06,
        "probe_hierarchy_subgoal_reachability_l1": 0.1,
        "probe_hierarchy_cem_model_bias_l1": 0.02,
    }
    (run / "probe_eval_balanced_v4.json").write_text(json.dumps(result))


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
        "probe_attention_entropy": 0.4 + 0.01 * step,
        "rollout_error_invalid_auroc": 0.6,
    }
    if loss is not None:
        record["train_loss"] = loss
    return record
