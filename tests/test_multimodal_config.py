from __future__ import annotations

from pathlib import Path

from cytof_archetypes.multimodal.config import DEFAULT_MULTIMODAL_CONFIG, deep_update, resolve_multimodal_paths


def test_default_multimodal_config_has_two_modalities() -> None:
    assert len(DEFAULT_MULTIMODAL_CONFIG["modalities"]) >= 2


def test_deep_update_overrides_nested_keys() -> None:
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    merged = deep_update(base, {"a": {"y": 99}})
    assert merged["a"]["x"] == 1
    assert merged["a"]["y"] == 99
    assert merged["b"] == 3


def test_resolve_multimodal_paths() -> None:
    config = {
        "modalities": [
            {"name": "m1", "input_path": "data/m1.csv"},
            {"name": "m2", "input_path": "data/m2.csv"},
        ],
        "alignment": {"cell_pairs_path": "data/pairs.csv"},
        "output": {"base_dir": "outputs/mm"},
    }
    root = Path("/tmp/multimodal_cfg")
    resolved = resolve_multimodal_paths(config, config_dir=root)
    resolved_root = str(root.resolve())
    assert resolved["modalities"][0]["input_path"].startswith(resolved_root)
    assert resolved["alignment"]["cell_pairs_path"].startswith(resolved_root)
    assert resolved["output"]["base_dir"].startswith(resolved_root)
