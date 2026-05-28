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
