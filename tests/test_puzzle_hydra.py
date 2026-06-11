from pathlib import Path

from hydra import compose, initialize_config_dir

from puzzle_jepa.train import run_smoke_experiment


def test_hydra_jepa_sudoku_smoke_config_runs():
    repo_root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(version_base=None, config_dir=str(repo_root / "configs" / "puzzle")):
        cfg = compose(config_name="jepa_sudoku_smoke")
    metrics = run_smoke_experiment(cfg)
    assert metrics["task"] == "sudoku"
    assert metrics["model_type"] == "jepa"
    assert metrics["loss"] >= 0.0


def test_hydra_trm_sudoku_smoke_config_runs():
    repo_root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(version_base=None, config_dir=str(repo_root / "configs" / "puzzle")):
        cfg = compose(config_name="trm_sudoku_smoke")
    metrics = run_smoke_experiment(cfg)
    assert metrics["task"] == "sudoku"
    assert metrics["model_type"] == "trm"
    assert metrics["loss"] > 0.0


def test_hydra_grid1_curriculum_configs_compose():
    repo_root = Path(__file__).resolve().parents[1]
    names = [
        "grid1_sudoku_jepa_5m_oracle",
        "grid1_sudoku_jepa_5m_mix70_30",
        "grid1_sudoku_jepa_5m_mix50_50",
        "grid1_maze_jepa_5m_oracle",
        "grid1_maze_jepa_5m_mix70_30",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(repo_root / "configs" / "puzzle")):
        configs = [compose(config_name=name) for name in names]
    assert [float(cfg.data.oracle_probability) for cfg in configs] == [1.0, 0.7, 0.5, 1.0, 0.7]


def test_hydra_grid4a_goal_energy_hierarchy_configs_compose():
    repo_root = Path(__file__).resolve().parents[1]
    names = [
        "grid4a_sudoku_jepa_5m_goal_energy_cem_l1",
        "grid4a_sudoku_jepa_5m_goal_energy_cem_l2",
        "grid4a_sudoku_jepa_5m_goal_energy_cem_l3",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(repo_root / "configs" / "puzzle")):
        configs = [compose(config_name=name) for name in names]
    assert [int(cfg.model.hierarchy_levels) for cfg in configs] == [1, 2, 3]
    assert [int(cfg.model.hierarchy_span) for cfg in configs] == [9, 9, 3]
    assert all(bool(cfg.model.use_cls_token) for cfg in configs)
    assert all(bool(cfg.model.use_goal_energy_head) for cfg in configs)


def test_hydra_grid4u_global_mlp_configs_compose():
    repo_root = Path(__file__).resolve().parents[1]
    names = [
        "grid4u_sudoku_global_mlp_l1",
        "grid4u_sudoku_global_mlp_l2",
        "grid4u_sudoku_global_mlp_l3",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(repo_root / "configs" / "puzzle")):
        configs = [compose(config_name=name) for name in names]
    assert [str(cfg.model.encoder_type) for cfg in configs] == ["global_mlp", "global_mlp", "global_mlp"]
    assert [int(cfg.model.hierarchy_levels) for cfg in configs] == [1, 2, 3]
    assert [int(cfg.model.action_embedding_dim) for cfg in configs] == [32, 32, 32]
    assert all(not bool(cfg.model.use_cls_token) for cfg in configs)


def test_hydra_grid4x_mixed_rollout_configs_compose():
    repo_root = Path(__file__).resolve().parents[1]
    names = [
        "grid4x_sudoku_global_mlp_l1_mixed_rollout_k2",
        "grid4x_sudoku_global_mlp_l1_mixed_rollout_k4",
        "grid4x_sudoku_global_mlp_l2_mixed_hierarchy_span4",
        "grid4x_sudoku_global_mlp_l3_mixed_hierarchy_span4",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(repo_root / "configs" / "puzzle")):
        configs = [compose(config_name=name) for name in names]
    assert [str(cfg.model.encoder_type) for cfg in configs] == ["global_mlp"] * 4
    assert [int(cfg.training.rollout_steps) for cfg in configs] == [2, 4, 2, 2]
    assert [float(cfg.training.rollout_oracle_probability) for cfg in configs] == [0.5, 0.5, 0.5, 0.5]
    assert [int(cfg.model.hierarchy_levels) for cfg in configs] == [1, 1, 2, 3]
    assert [float(cfg.training.hierarchy_weight) for cfg in configs] == [0.0, 0.0, 1.0, 1.0]
    assert [int(cfg.training.hierarchy_rollout_steps) for cfg in configs] == [1, 1, 4, 16]
    assert [float(cfg.training.get("hierarchy_oracle_probability", 1.0)) for cfg in configs] == [1.0, 1.0, 0.5, 0.5]
