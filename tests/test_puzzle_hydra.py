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


def test_hydra_grid5_sigreg_config_composes():
    repo_root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(version_base=None, config_dir=str(repo_root / "configs" / "puzzle")):
        cfg = compose(config_name="grid5_sudoku_sigreg")
    assert cfg.model.sigreg_weight == 1.0
    assert cfg.training.goal_energy_weight == 1.0
    assert cfg.training.rollout_oracle_probability == 0.5
