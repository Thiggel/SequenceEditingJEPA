from pathlib import Path

from seq_edit_jepa.models import SequenceEditJEPA
from seq_edit_jepa.train.train_one_step import run_experiment


def test_smoke_experiment_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("SEQ_EDIT_JEPA_WORK_ROOT", str(tmp_path))
    metrics = run_experiment("configs/smoke_lano_mask.yaml")
    assert "eval/loss" in metrics
    assert "eval/token_accuracy" in metrics
    assert "eval/lano/grammar_validity" in metrics
    run_dir = Path(tmp_path, "runs", "smoke_lano_mask")
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "checkpoint.pt").exists()
    assert (run_dir / "model" / "config.json").exists()
    assert any((run_dir / "model").glob("*.safetensors")) or (run_dir / "model" / "pytorch_model.bin").exists()
    assert (run_dir / "tokenizer" / "vocab.json").exists()
    loaded = SequenceEditJEPA.from_pretrained(run_dir / "model")
    assert loaded.config.model_type == "seq_edit_jepa"
