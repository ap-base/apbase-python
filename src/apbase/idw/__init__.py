"""High-level inverse distance weighting API.

This package exports the public class used to run IDW interpolation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ._interpolation import IDW

__all__ = ["IDW", "idw"]


def idw(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    targets: ArrayLike,
    *,
    radius: float | None = None,
    power: float = 2.0,
    max_neighbors: int = 40,
    min_neighbors: int = 3,
) -> np.ndarray:
    """Fit IDW from source data and estimate ``targets`` values."""
    return IDW(
        radius=radius,
        power=power,
        max_neighbors=max_neighbors,
        min_neighbors=min_neighbors,
    ).fit(x, y, z).interpolate(targets)
