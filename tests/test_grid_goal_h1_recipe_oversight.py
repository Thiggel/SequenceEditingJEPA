import json
import importlib.util
import sys
from pathlib import Path


_SPEC = importlib.util.spec_from_file_location(
    "grid_goal_h1_recipe",
    Path(__file__).resolve().parents[1] / "scripts" / "oversight" / "grid_goal_h1_recipe.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
summarize_h1_recipe = _MODULE.summarize_h1_recipe


def _write_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")


def _touch_checkpoint(run_root, variant):
    root = run_root / f"grid_goal_h1_recipe_{variant}"
    root.mkdir(parents=True, exist_ok=True)
    (root / "checkpoint.pt").write_bytes(b"placeholder")


def test_h1_recipe_oversight_selects_best_action_and_dynamics_from_local_oracle_rows(tmp_path):
    run_root = tmp_path / "grid_goal_h1_recipe"
    for variant in ("anchor_h1", "action_old_local_value", "dynamics_affected_context"):
        _touch_checkpoint(run_root, variant)

    base = {
        "planner": "mpc_beam",
        "transition_mode": "latent_rollout",
        "score_mode": "oracle_goal_changed_cell_raw_euclidean_distance",
        "beam_depth": 16,
        "solve_rate": 0.0,
        "solved": 0,
        "remaining_hamming_mean": 12.0,
    }
    _write_row(run_root / "grid_goal_h1_recipe_anchor_h1" / "planner_eval_h1_recipe" / "planner_matrix.jsonl", base)
    _write_row(
        run_root / "grid_goal_h1_recipe_action_old_local_value" / "planner_eval_h1_recipe" / "planner_matrix.jsonl",
        {**base, "remaining_hamming_mean": 3.0},
    )
    _write_row(
        run_root / "grid_goal_h1_recipe_dynamics_affected_context" / "planner_eval_h1_recipe" / "planner_matrix.jsonl",
        {**base, "score_mode": "oracle_goal_affected_context_raw_euclidean_distance", "remaining_hamming_mean": 2.0},
    )

    summary = summarize_h1_recipe(run_root)

    assert summary["best_action_variant"] == "action_old_local_value"
    assert summary["best_action_conditioning"] == "old_local_value"
    assert summary["best_dynamics_variant"] == "dynamics_affected_context"
    assert summary["best_dynamics_weighting"] == "affected_context"
    assert summary["best_local_oracle"]["remaining_hamming_mean"] == 2.0
