"""Atomic persistence helpers for baseline progress artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd


def atomic_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    """Replace a CSV only after the temporary file is complete."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def atomic_npz(
    arrays: Mapping[str, np.ndarray],
    path: str | Path,
) -> Path:
    """Replace a compressed NumPy archive atomically."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
