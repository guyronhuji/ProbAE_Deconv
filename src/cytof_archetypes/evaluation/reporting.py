from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)


def write_markdown(path: str | Path, text: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")


def write_json(path: str | Path, payload: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_environment_log(path: str | Path, extra: dict | None = None) -> None:
    payload = {
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }
    if extra:
        payload.update(extra)
    write_json(path, payload)
