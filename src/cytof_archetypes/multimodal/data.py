from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from cytof_archetypes.preprocessing import MarkerPreprocessor


@dataclass
class PreparedModalitySplit:
    x_encoder: np.ndarray
    x_target: np.ndarray
    library_size: np.ndarray
    cell_ids: np.ndarray
    sample_ids: np.ndarray
    labels: np.ndarray | None
    obs: pd.DataFrame


@dataclass
class PreparedModality:
    name: str
    marker_names: list[str]
    decoder_family: str
    preprocessor_state: dict[str, Any] | None
    train: PreparedModalitySplit
    val: PreparedModalitySplit
    test: PreparedModalitySplit


@dataclass
class CellPairSplit:
    left_modality: str
    right_modality: str
    left_indices: np.ndarray
    right_indices: np.ndarray


@dataclass
class PreparedMultimodalData:
    modalities: dict[str, PreparedModality]
    modality_order: list[str]
    shared_samples: dict[str, np.ndarray]
    pair_indices: dict[str, CellPairSplit] | None


@dataclass
class _LoadedModality:
    name: str
    marker_names: list[str]
    decoder_family: str
    cell_id_column: str
    sample_id_column: str
    label_column: str | None
    frame: pd.DataFrame
    x_raw: np.ndarray


def _is_count_decoder(decoder_family: str) -> bool:
    return str(decoder_family).lower() in {"nb", "beta_binomial"}


def _compute_library_size(x_target: np.ndarray) -> np.ndarray:
    lib = np.sum(x_target, axis=1).astype(np.float32)
    return np.clip(lib, 1e-8, None)


def _build_encoder_input(x_target: np.ndarray, encoder_input: str) -> np.ndarray:
    mode = str(encoder_input).lower()
    if mode == "raw_counts":
        return x_target.astype(np.float32)
    if mode == "log1p_normalized":
        library_size = _compute_library_size(x_target)
        normalized = x_target / library_size[:, None] * 1.0e4
        return np.log1p(normalized).astype(np.float32)
    raise ValueError(f"Unsupported data.encoder_input mode: {encoder_input}")


def _read_table(input_path: str | Path) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing modality input path: {path}")
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".tsv"}:
        return pd.read_csv(path, sep="\t")
    if suffix in {".h5ad"}:
        try:
            import anndata as ad
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("anndata is required to load .h5ad modality files") from exc
        adata = ad.read_h5ad(path)
        frame = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
        obs = adata.obs.copy()
        obs.index = obs.index.astype(str)
        frame.index = frame.index.astype(str)
        merged = pd.concat([obs, frame], axis=1)
        return merged.reset_index(drop=False).rename(columns={"index": "_obs_index"})
    raise ValueError(f"Unsupported modality file format: {path}")


def _load_markers_from_path(path: str | Path) -> list[str]:
    marker_path = Path(path)
    if marker_path.suffix.lower() in {".txt", ".tsv", ".csv"}:
        if marker_path.suffix.lower() == ".txt":
            markers = [line.strip() for line in marker_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            return markers
        frame = pd.read_csv(marker_path)
        if frame.shape[1] == 0:
            return []
        return [str(v) for v in frame.iloc[:, 0].dropna().astype(str).tolist()]
    raise ValueError(f"Unsupported marker_columns_path format: {marker_path}")


def _resolve_marker_columns(frame: pd.DataFrame, modality_cfg: dict[str, Any]) -> list[str]:
    marker_columns = modality_cfg.get("marker_columns")
    marker_columns_path = modality_cfg.get("marker_columns_path")
    if marker_columns_path:
        marker_columns = _load_markers_from_path(marker_columns_path)

    if marker_columns is not None:
        markers = [str(col) for col in marker_columns if str(col) in frame.columns]
        if not markers:
            raise ValueError(
                f"No requested marker columns were found for modality '{modality_cfg.get('name', '<unnamed>')}'."
            )
        return markers

    reserved = {
        str(modality_cfg.get("cell_id_column", "cell_id")),
        str(modality_cfg.get("sample_id_column", "sample_id")),
    }
    label_column = modality_cfg.get("label_column")
    if label_column:
        reserved.add(str(label_column))
    obs_columns = modality_cfg.get("obs_columns")
    if obs_columns:
        reserved.update(str(col) for col in obs_columns)

    numeric_cols = [col for col in frame.columns if col not in reserved and pd.api.types.is_numeric_dtype(frame[col])]
    if not numeric_cols:
        raise ValueError(
            f"Could not infer marker columns for modality '{modality_cfg.get('name', '<unnamed>')}'. "
            "Set marker_columns explicitly in config."
        )
    return [str(col) for col in numeric_cols]


def _sanitize_frame(frame: pd.DataFrame, modality_cfg: dict[str, Any], marker_names: list[str]) -> pd.DataFrame:
    cell_col = str(modality_cfg.get("cell_id_column", "cell_id"))
    sample_col = str(modality_cfg.get("sample_id_column", "sample_id"))
    label_col = modality_cfg.get("label_column")

    if cell_col not in frame.columns:
        raise ValueError(f"Missing cell_id column '{cell_col}' in modality '{modality_cfg.get('name')}'.")
    if sample_col not in frame.columns:
        raise ValueError(f"Missing sample_id column '{sample_col}' in modality '{modality_cfg.get('name')}'.")

    keep_cols = [cell_col, sample_col]
    if label_col and str(label_col) in frame.columns:
        keep_cols.append(str(label_col))

    obs_columns = modality_cfg.get("obs_columns")
    if obs_columns:
        keep_cols.extend(str(col) for col in obs_columns if str(col) in frame.columns)

    marker_cols = [col for col in marker_names if col in frame.columns]
    working = frame[keep_cols + marker_cols].copy()

    working = working.dropna(subset=[cell_col, sample_col])
    marker_nonempty = [col for col in marker_cols if not working[col].isna().all()]
    if not marker_nonempty:
        raise ValueError(f"All selected marker columns are empty for modality '{modality_cfg.get('name')}'.")

    working = working[keep_cols + marker_nonempty]
    for col in marker_nonempty:
        working[col] = pd.to_numeric(working[col], errors="coerce")
    working = working.dropna(subset=marker_nonempty, how="all")
    working[marker_nonempty] = working[marker_nonempty].fillna(0.0)

    working[cell_col] = working[cell_col].astype(str)
    duplicates = working[cell_col].duplicated(keep=False)
    if duplicates.any():
        suffix = working.groupby(cell_col).cumcount().astype(str)
        working.loc[duplicates, cell_col] = working.loc[duplicates, cell_col] + "__" + suffix[duplicates]

    working[sample_col] = working[sample_col].astype(str)
    if label_col and str(label_col) in working.columns:
        working[str(label_col)] = working[str(label_col)].astype(str)
    return working


def _load_modality(modality_cfg: dict[str, Any]) -> _LoadedModality:
    frame = _read_table(modality_cfg["input_path"])
    marker_names = _resolve_marker_columns(frame, modality_cfg)
    cleaned = _sanitize_frame(frame, modality_cfg=modality_cfg, marker_names=marker_names)
    marker_names = [m for m in marker_names if m in cleaned.columns]

    x_raw = cleaned[marker_names].to_numpy(dtype=np.float32)
    return _LoadedModality(
        name=str(modality_cfg["name"]),
        marker_names=marker_names,
        decoder_family=str(modality_cfg.get("model", {}).get("decoder_family", "gaussian")).lower(),
        cell_id_column=str(modality_cfg.get("cell_id_column", "cell_id")),
        sample_id_column=str(modality_cfg.get("sample_id_column", "sample_id")),
        label_column=str(modality_cfg["label_column"]) if modality_cfg.get("label_column") else None,
        frame=cleaned,
        x_raw=x_raw,
    )


def _safe_split(items: np.ndarray, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if len(items) == 0 or test_fraction <= 0.0:
        return items, np.asarray([], dtype=items.dtype)
    if len(items) < 2:
        return items, np.asarray([], dtype=items.dtype)
    train_items, test_items = train_test_split(items, test_size=test_fraction, random_state=seed)
    return np.asarray(train_items), np.asarray(test_items)


def _split_sample_ids(
    loaded: dict[str, _LoadedModality],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_samples: list[np.ndarray] = []
    for modality in loaded.values():
        all_samples.append(modality.frame[modality.sample_id_column].astype(str).unique())
    unique_samples = np.unique(np.concatenate(all_samples, axis=0)) if all_samples else np.asarray([], dtype=object)

    train_val, test_ids = _safe_split(unique_samples, test_fraction=test_fraction, seed=seed)
    val_ratio = val_fraction / max(1.0 - test_fraction, 1.0e-8)
    train_ids, val_ids = _safe_split(train_val, test_fraction=val_ratio, seed=seed)
    return train_ids.astype(str), val_ids.astype(str), test_ids.astype(str)


def _split_cell_indices(
    n_rows: int,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_idx = np.arange(n_rows)
    train_val, test_idx = _safe_split(all_idx, test_fraction=test_fraction, seed=seed)
    val_ratio = val_fraction / max(1.0 - test_fraction, 1.0e-8)
    train_idx, val_idx = _safe_split(train_val, test_fraction=val_ratio, seed=seed)
    return train_idx.astype(int), val_idx.astype(int), test_idx.astype(int)


def _build_split(
    modality: _LoadedModality,
    modality_cfg: dict[str, Any],
    split_idx: np.ndarray,
    data_cfg: dict[str, Any],
) -> PreparedModalitySplit:
    frame = modality.frame.iloc[split_idx].reset_index(drop=True)
    x_raw = modality.x_raw[split_idx]
    decoder_family = modality.decoder_family

    if _is_count_decoder(decoder_family):
        x_target = np.clip(x_raw.astype(np.float32), 0.0, None)
        x_encoder = _build_encoder_input(x_target, data_cfg.get("encoder_input", "log1p_normalized"))

        use_observed = bool(modality_cfg.get("model", {}).get("use_observed_library_size", True))
        if use_observed:
            library_size = _compute_library_size(x_target)
        else:
            size_factor_key = modality_cfg.get("model", {}).get("size_factor_key")
            if size_factor_key is None:
                library_size = _compute_library_size(x_target)
            elif size_factor_key not in frame.columns:
                raise ValueError(
                    f"Requested size_factor_key='{size_factor_key}' for modality '{modality.name}' but column is missing."
                )
            else:
                library_size = frame[size_factor_key].to_numpy(dtype=np.float32)
                library_size = np.clip(library_size, 1e-8, None)
        return PreparedModalitySplit(
            x_encoder=x_encoder.astype(np.float32),
            x_target=x_target.astype(np.float32),
            library_size=library_size.astype(np.float32),
            cell_ids=frame[modality.cell_id_column].to_numpy(dtype=str),
            sample_ids=frame[modality.sample_id_column].to_numpy(dtype=str),
            labels=frame[modality.label_column].to_numpy(dtype=str) if modality.label_column and modality.label_column in frame.columns else None,
            obs=frame,
        )

    pre_cfg = modality_cfg.get("preprocessing", {})
    preprocessor = MarkerPreprocessor(
        transform=str(pre_cfg.get("transform", "none")),
        arcsinh_cofactor=float(pre_cfg.get("arcsinh_cofactor", 5.0)),
        normalization=str(pre_cfg.get("normalization", "zscore")),
        clip_min=pre_cfg.get("clip_min"),
        clip_max=pre_cfg.get("clip_max"),
    )
    # Placeholder, fitted in prepare_multimodal_data on full train split.
    del preprocessor
    return PreparedModalitySplit(
        x_encoder=x_raw.astype(np.float32),
        x_target=x_raw.astype(np.float32),
        library_size=_compute_library_size(np.clip(x_raw, 0.0, None)),
        cell_ids=frame[modality.cell_id_column].to_numpy(dtype=str),
        sample_ids=frame[modality.sample_id_column].to_numpy(dtype=str),
        labels=frame[modality.label_column].to_numpy(dtype=str) if modality.label_column and modality.label_column in frame.columns else None,
        obs=frame,
    )


def _fit_transform_gaussian_modality(
    train_split: PreparedModalitySplit,
    val_split: PreparedModalitySplit,
    test_split: PreparedModalitySplit,
    modality_cfg: dict[str, Any],
) -> tuple[PreparedModalitySplit, PreparedModalitySplit, PreparedModalitySplit, dict[str, Any]]:
    pre_cfg = modality_cfg.get("preprocessing", {})
    preprocessor = MarkerPreprocessor(
        transform=str(pre_cfg.get("transform", "none")),
        arcsinh_cofactor=float(pre_cfg.get("arcsinh_cofactor", 5.0)),
        normalization=str(pre_cfg.get("normalization", "zscore")),
        clip_min=pre_cfg.get("clip_min"),
        clip_max=pre_cfg.get("clip_max"),
    )
    train_x = preprocessor.fit_transform(train_split.x_target)
    val_x = preprocessor.transform_array(val_split.x_target) if len(val_split.x_target) else val_split.x_target
    test_x = preprocessor.transform_array(test_split.x_target) if len(test_split.x_target) else test_split.x_target

    train_out = PreparedModalitySplit(
        x_encoder=train_x,
        x_target=train_x,
        library_size=_compute_library_size(np.clip(train_x, 0.0, None)),
        cell_ids=train_split.cell_ids,
        sample_ids=train_split.sample_ids,
        labels=train_split.labels,
        obs=train_split.obs,
    )
    val_out = PreparedModalitySplit(
        x_encoder=val_x,
        x_target=val_x,
        library_size=_compute_library_size(np.clip(val_x, 0.0, None)) if len(val_x) else np.zeros((0,), dtype=np.float32),
        cell_ids=val_split.cell_ids,
        sample_ids=val_split.sample_ids,
        labels=val_split.labels,
        obs=val_split.obs,
    )
    test_out = PreparedModalitySplit(
        x_encoder=test_x,
        x_target=test_x,
        library_size=_compute_library_size(np.clip(test_x, 0.0, None)) if len(test_x) else np.zeros((0,), dtype=np.float32),
        cell_ids=test_split.cell_ids,
        sample_ids=test_split.sample_ids,
        labels=test_split.labels,
        obs=test_split.obs,
    )
    return train_out, val_out, test_out, preprocessor.state_dict()


def _build_pair_indices(
    prepared: dict[str, PreparedModality],
    alignment_cfg: dict[str, Any],
) -> dict[str, CellPairSplit] | None:
    path = alignment_cfg.get("cell_pairs_path")
    if path is None:
        return None

    pairs = pd.read_csv(path)
    modalities = list(prepared.keys())
    if len(modalities) < 2:
        return None

    left_modality = str(alignment_cfg.get("left_modality") or modalities[0])
    right_modality = str(alignment_cfg.get("right_modality") or modalities[1])
    if left_modality not in prepared or right_modality not in prepared:
        raise ValueError("Alignment modalities must be present in loaded modalities.")

    left_col = str(alignment_cfg.get("left_cell_id_column", "left_cell_id"))
    right_col = str(alignment_cfg.get("right_cell_id_column", "right_cell_id"))
    if left_col not in pairs.columns or right_col not in pairs.columns:
        raise ValueError(
            f"Cell pair table must include columns '{left_col}' and '{right_col}'."
        )

    out: dict[str, CellPairSplit] = {}
    for split_name in ("train", "val", "test"):
        left_split = getattr(prepared[left_modality], split_name)
        right_split = getattr(prepared[right_modality], split_name)
        left_map = {str(cell_id): idx for idx, cell_id in enumerate(left_split.cell_ids)}
        right_map = {str(cell_id): idx for idx, cell_id in enumerate(right_split.cell_ids)}

        left_indices: list[int] = []
        right_indices: list[int] = []
        for _, row in pairs.iterrows():
            left_id = str(row[left_col])
            right_id = str(row[right_col])
            if left_id in left_map and right_id in right_map:
                left_indices.append(int(left_map[left_id]))
                right_indices.append(int(right_map[right_id]))

        out[split_name] = CellPairSplit(
            left_modality=left_modality,
            right_modality=right_modality,
            left_indices=np.asarray(left_indices, dtype=np.int64),
            right_indices=np.asarray(right_indices, dtype=np.int64),
        )
    return out


def _shared_samples(prepared: dict[str, PreparedModality]) -> dict[str, np.ndarray]:
    split_shared: dict[str, np.ndarray] = {}
    for split_name in ("train", "val", "test"):
        per_modality: list[np.ndarray] = []
        for modality in prepared.values():
            per_modality.append(np.unique(getattr(modality, split_name).sample_ids))
        if not per_modality:
            split_shared[split_name] = np.asarray([], dtype=str)
            continue
        common = per_modality[0]
        for arr in per_modality[1:]:
            common = np.intersect1d(common, arr)
        split_shared[split_name] = common.astype(str)
    return split_shared


def prepare_multimodal_data(config: dict[str, Any]) -> PreparedMultimodalData:
    modalities_cfg = config.get("modalities", [])
    if len(modalities_cfg) < 2:
        raise ValueError("Multimodal config requires at least two modalities.")

    loaded: dict[str, _LoadedModality] = {}
    for modality_cfg in modalities_cfg:
        loaded_modality = _load_modality(modality_cfg)
        if loaded_modality.name in loaded:
            raise ValueError(f"Duplicate modality name detected: {loaded_modality.name}")
        loaded[loaded_modality.name] = loaded_modality

    split_cfg = config.get("split", {})
    val_fraction = float(split_cfg.get("val_fraction", 0.15))
    test_fraction = float(split_cfg.get("test_fraction", 0.15))
    split_level = str(split_cfg.get("level", "sample")).lower()
    seed = int(config.get("seed", 42))

    split_indices: dict[str, dict[str, np.ndarray]] = {}
    if split_level == "sample":
        train_samples, val_samples, test_samples = _split_sample_ids(
            loaded,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
        )
        train_set = set(train_samples.tolist())
        val_set = set(val_samples.tolist())
        test_set = set(test_samples.tolist())
        for name, modality in loaded.items():
            sample_ids = modality.frame[modality.sample_id_column].astype(str).to_numpy()
            split_indices[name] = {
                "train": np.where(np.isin(sample_ids, list(train_set)))[0].astype(int),
                "val": np.where(np.isin(sample_ids, list(val_set)))[0].astype(int),
                "test": np.where(np.isin(sample_ids, list(test_set)))[0].astype(int),
            }
    elif split_level == "cell":
        for idx, (name, modality) in enumerate(loaded.items()):
            train_idx, val_idx, test_idx = _split_cell_indices(
                n_rows=modality.x_raw.shape[0],
                val_fraction=val_fraction,
                test_fraction=test_fraction,
                seed=seed + idx,
            )
            split_indices[name] = {
                "train": train_idx,
                "val": val_idx,
                "test": test_idx,
            }
    else:
        raise ValueError("split.level must be either 'sample' or 'cell'.")

    prepared: dict[str, PreparedModality] = {}
    data_cfg = config.get("data", {})
    modality_cfg_map = {str(mod_cfg["name"]): mod_cfg for mod_cfg in modalities_cfg}

    for modality_name, modality in loaded.items():
        modality_cfg = modality_cfg_map[modality_name]
        idx = split_indices[modality_name]
        train = _build_split(modality, modality_cfg=modality_cfg, split_idx=idx["train"], data_cfg=data_cfg)
        val = _build_split(modality, modality_cfg=modality_cfg, split_idx=idx["val"], data_cfg=data_cfg)
        test = _build_split(modality, modality_cfg=modality_cfg, split_idx=idx["test"], data_cfg=data_cfg)

        preprocessor_state = None
        if not _is_count_decoder(modality.decoder_family):
            train, val, test, preprocessor_state = _fit_transform_gaussian_modality(
                train_split=train,
                val_split=val,
                test_split=test,
                modality_cfg=modality_cfg,
            )

        prepared[modality_name] = PreparedModality(
            name=modality_name,
            marker_names=modality.marker_names,
            decoder_family=modality.decoder_family,
            preprocessor_state=preprocessor_state,
            train=train,
            val=val,
            test=test,
        )

    shared_samples = _shared_samples(prepared)
    pair_indices = _build_pair_indices(prepared, alignment_cfg=config.get("alignment", {}))

    return PreparedMultimodalData(
        modalities=prepared,
        modality_order=list(loaded.keys()),
        shared_samples=shared_samples,
        pair_indices=pair_indices,
    )
