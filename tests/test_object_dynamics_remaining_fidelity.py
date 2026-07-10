from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from puzzle_jepa.object_dynamics.batching import RELATION_NAMES
from puzzle_jepa.object_dynamics.generator import ObjectDynamicsGenerator, ObjectDynamicsSpec, TRAJECTORY_KINDS
from puzzle_jepa.object_dynamics.model import ObjectDynamicsJEPA


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.xfail(
    strict=True,
    reason="Two train lengths cannot establish the small/medium/large probe-saturation curve required by the prestage plan.",
)
def test_prestage_has_three_well_separated_train_lengths() -> None:
    config = OmegaConf.load(ROOT / "configs/object_dynamics/sweep/prestage.yaml")
    train_lengths = list(config.max_steps)
    assert len(train_lengths) >= 3
    assert max(train_lengths) >= 10 * min(train_lengths)


@pytest.mark.xfail(
    strict=True,
    reason="The project invariant requires full-grid partners for every single-CLS Delta-JEPA row.",
)
def test_delta_jepa_rows_have_full_grid_partners() -> None:
    assert (ROOT / "configs/object_dynamics/model/grid128_r8.yaml").exists()
    assert (ROOT / "configs/object_dynamics/model/h_grid128_h8.yaml").exists()
    script = (ROOT / "scripts/experiments/submit_object_dynamics_phase1.sh").read_text()
    assert "grid128_r8" in script
    assert "h_grid128_h8" in script


@pytest.mark.xfail(
    strict=True,
    reason="The compression claim needs a full-grid latent baseline in Phase 1, not only for Delta-JEPA rows.",
)
def test_phase1_includes_full_grid_latent_baseline() -> None:
    assert (ROOT / "configs/object_dynamics/model/grid128_r8.yaml").exists()
    script = (ROOT / "scripts/experiments/submit_object_dynamics_phase1.sh").read_text()
    assert "grid128_r8" in script


@pytest.mark.xfail(
    strict=True,
    reason="Object LDAD decodes adjacent edits only; Delta-JEPA also defines long-horizon displacement-to-action-sequence decoding.",
)
def test_object_ldad_supports_long_horizon_action_sequence_decoding() -> None:
    model = ObjectDynamicsJEPA(
        grid_size=8,
        d_model=32,
        encoder_layers=1,
        encoder_heads=4,
        rollout_horizon=4,
        target_mode="shared",
        ldad_weight=0.1,
    )
    assert callable(model.decode_action_sequence_from_displacement)


def test_generator_has_pure_random_edit_control() -> None:
    assert "random_off_manifold" in TRAJECTORY_KINDS
    assert (ROOT / "configs/object_dynamics/data/random_off_manifold.yaml").exists()


@pytest.mark.xfail(
    strict=True,
    reason="The current hierarchy is an endpoint training loss, not the HWM macro-action planning/evaluation stack.",
)
def test_hierarchy_exposes_high_and_low_level_planning_rollouts() -> None:
    model = ObjectDynamicsJEPA(
        grid_size=8,
        d_model=32,
        encoder_layers=1,
        encoder_heads=4,
        rollout_horizon=1,
        hierarchy_horizon=4,
    )
    assert callable(model.rollout_high_level)
    assert callable(model.plan_macro_actions)


@pytest.mark.xfail(
    strict=True,
    reason="Scene metadata still lacks explicit part masks and the planned inside relation.",
)
def test_scene_metadata_covers_parts_and_inside_relations() -> None:
    scene = ObjectDynamicsGenerator(ObjectDynamicsSpec()).sample_scene(np.random.default_rng(5))
    assert all(hasattr(obj, "parts") for obj in scene.objects)
    assert "inside" in RELATION_NAMES


@pytest.mark.xfail(
    strict=True,
    reason="The encoder does not expose CLS attention maps or current-object/incomplete-object IoU metrics.",
)
def test_probe_suite_includes_attention_evidence() -> None:
    probe_source = (ROOT / "puzzle_jepa/object_dynamics/probes.py").read_text()
    assert "attention_current_object_iou" in probe_source


@pytest.mark.xfail(
    strict=True,
    reason="A reconstruction-trained encoder baseline is required to attribute object factors specifically to JEPA training.",
)
def test_phase_sweep_includes_reconstruction_trained_encoder_baseline() -> None:
    probe_source = (ROOT / "puzzle_jepa/object_dynamics/probes.py").read_text()
    assert "autoencoder_baseline" in probe_source


@pytest.mark.xfail(
    strict=True,
    reason="The plan calls for a small-MLP upper bound and correction-type hierarchy labels after the linear probes.",
)
def test_probe_suite_includes_nonlinear_upper_bound_and_correction_chunks() -> None:
    probe_source = (ROOT / "puzzle_jepa/object_dynamics/probes.py").read_text()
    assert "probe_mlp_" in probe_source
    assert "chunk_correction" in probe_source


@pytest.mark.xfail(
    strict=True,
    reason="The phase launcher currently runs one seed; confirmatory comparisons require at least three independent seeds.",
)
def test_phase_sweep_runs_multiple_seeds() -> None:
    script = (ROOT / "scripts/experiments/submit_object_dynamics_phase1.sh").read_text()
    assert "SEEDS=(" in script
    assert script.count("SEED=") >= 1
