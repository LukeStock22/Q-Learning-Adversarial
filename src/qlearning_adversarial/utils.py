"""Shared helpers."""

from __future__ import annotations

import random
from pathlib import Path


def seed_everything(seed: int) -> None:
    """Seed Python (and NumPy if present) for reproducibility."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        # Numpy isn't required for all paths; ignore if unavailable.
        pass


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it doesn't exist and return its Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
