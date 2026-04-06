from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(payload: dict[str, Any], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def next_run_dir(base_dir: str | Path) -> Path:
    base = ensure_dir(base_dir)
    existing = []
    for path in base.glob("run_*"):
        suffix = path.name.replace("run_", "")
        if suffix.isdigit():
            existing.append(int(suffix))
    next_idx = (max(existing) + 1) if existing else 1
    run_dir = base / f"run_{next_idx:03d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir
