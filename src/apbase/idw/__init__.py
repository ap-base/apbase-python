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
    """Fit IDW from source data and estimate target values.

    Parameters
    ----------
    x, y, z : array_like
        Source coordinates and values.
    targets : array_like
        Target coordinates accepted by :meth:`apbase.idw.IDW.interpolate`.
    radius : float or None, default None
        Local search radius. If ``None``, derive ``range / 3`` from a fitted
        variogram.
    power : float, default 2.0
        IDW distance exponent.
    max_neighbors, min_neighbors : int
        Neighbor bounds used by the local search.

    Returns
    -------
    numpy.ndarray
        One interpolated value per target coordinate.
    """
    return IDW(
        radius=radius,
        power=power,
        max_neighbors=max_neighbors,
        min_neighbors=min_neighbors,
    ).fit(x, y, z).interpolate(targets)
