"""
Prepare breast CyTOF dataset: merge normalized_not_scaled parquet files → h5ad.

Usage:
    python scripts/prepare_breast_dataset.py \
        --input-dir /path/to/parquets \
        --output data/breast_cytof_processed.h5ad

The script:
- Excludes .2 replicate samples (override with --include-dot2)
- Finds markers common to ALL included samples (avoids NaN columns)
- Excludes backbone markers H3, H4, H3.3
- Clips each marker to [0, 99.9th percentile]
- Stores sample_id in obs (no cell type labels)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


EXCLUDED_MARKERS = {"H3", "H4", "H3.3"}
SAMPLE_ID_REGEX = r"_([0-9]+(?:\.[0-9]+)?)$"


def prepare(
    input_dir: Path,
    output_path: Path,
    file_variant: str = "normalized_not_scaled",
    exclude_dot2: bool = True,
    clip_upper_pct: float = 99.9,
    overwrite: bool = False,
) -> None:
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError("anndata required: pip install anndata") from exc

    if output_path.exists() and not overwrite:
        print(f"Output already exists: {output_path}")
        print("Use --overwrite to regenerate.")
        return

    files = sorted(input_dir.glob(f"{file_variant}_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No '{file_variant}_*.parquet' files found in {input_dir}")
    print(f"Found {len(files)} parquet files.")

    # First pass: marker intersection across included samples
    common_markers: set[str] | None = None
    for fp in files:
        m = re.search(SAMPLE_ID_REGEX, fp.stem)
        sid = m.group(1) if m else fp.stem
        if exclude_dot2 and sid.endswith(".2"):
            continue
        df = pd.read_parquet(fp)
        cols = {
            c for c in df.columns
            if c not in EXCLUDED_MARKERS and pd.api.types.is_numeric_dtype(df[c])
        }
        common_markers = cols if common_markers is None else common_markers & cols

    if not common_markers:
        raise ValueError("No common numeric markers found across included samples.")

    marker_cols = sorted(common_markers)
    print(f"Common markers ({len(marker_cols)}): {marker_cols}")

    # Second pass: load with common markers only
    dfs = []
    for fp in files:
        m = re.search(SAMPLE_ID_REGEX, fp.stem)
        sid = m.group(1) if m else fp.stem
        if exclude_dot2 and sid.endswith(".2"):
            print(f"  Skipping {fp.name}  (replicate)")
            continue
        df = pd.read_parquet(fp)
        df["sample_id"] = sid
        dfs.append(df[marker_cols + ["sample_id"]])
        print(f"  Loaded  {fp.name}  ({len(df):,} cells)")

    combined = pd.concat(dfs, ignore_index=True)
    nan_count = combined[marker_cols].isna().sum().sum()
    print(f"\nTotal cells: {len(combined):,}  |  NaN in markers: {nan_count}")
    if nan_count > 0:
        raise ValueError(f"Unexpected NaN values in marker matrix ({nan_count} NaNs).")

    X = combined[marker_cols].values.astype("float32")
    upper = np.percentile(X, clip_upper_pct, axis=0)
    X = np.clip(X, 0, upper)

    import anndata as ad  # noqa: F811 (already imported above, re-bind for clarity)
    obs = pd.DataFrame(
        {"cell_id": [f"cell_{i}" for i in range(len(combined))],
         "sample_id": combined["sample_id"].values}
    )
    obs.index = obs["cell_id"].values

    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=marker_cols))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_path)
    print(f"\nSaved: {output_path}  ({adata.shape[0]:,} cells x {adata.shape[1]} markers)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True, type=Path,
                        help="Directory containing normalized_not_scaled_*.parquet files")
    parser.add_argument("--output", default="data/breast_cytof_processed.h5ad", type=Path,
                        help="Output h5ad path (default: data/breast_cytof_processed.h5ad)")
    parser.add_argument("--file-variant", default="normalized_not_scaled",
                        help="Parquet filename prefix (default: normalized_not_scaled)")
    parser.add_argument("--include-dot2", action="store_true",
                        help="Include .2 replicate samples (excluded by default)")
    parser.add_argument("--clip-upper-pct", type=float, default=99.9,
                        help="Upper percentile clip (default: 99.9)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output file")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"ERROR: --input-dir does not exist: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    prepare(
        input_dir=args.input_dir,
        output_path=args.output,
        file_variant=args.file_variant,
        exclude_dot2=not args.include_dot2,
        clip_upper_pct=args.clip_upper_pct,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
