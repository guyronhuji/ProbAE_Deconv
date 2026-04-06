from pathlib import Path

import numpy as np
import pandas as pd

from cytof_archetypes.datasets import load_levine32_bundle


def test_load_levine32_tabular(tmp_path: Path):
    rng = np.random.default_rng(0)
    n_cells = 120
    marker_cols = [f"m{i}" for i in range(32)]
    frame = pd.DataFrame(rng.normal(size=(n_cells, 32)), columns=marker_cols)
    frame["label"] = np.where(np.arange(n_cells) % 2 == 0, "A", "B")
    frame["cell_id"] = [f"c{i}" for i in range(n_cells)]
    data_path = tmp_path / "levine_like.csv"
    frame.to_csv(data_path, index=False)

    bundle = load_levine32_bundle(
        input_path=data_path,
        marker_columns=marker_cols,
        label_column="label",
        cell_id_column="cell_id",
        val_fraction=0.2,
        test_fraction=0.2,
        seed=42,
    )

    assert len(bundle.markers) == 32
    assert bundle.train.x.shape[1] == 32
    assert len(bundle.train.x) + len(bundle.val.x) + len(bundle.test.x) == n_cells
