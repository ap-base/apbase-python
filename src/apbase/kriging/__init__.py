"""High-level kriging API.

This package exports the public classes used to fit variogram models and run
local ordinary kriging. Target-grid generation lives in ``apbase.grid``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from apbase.variogram import Variogram

from ._interpolation import Kriging

__all__ = [
    "Kriging",
    "Variogram",
    "ordinary_kriging",
]


def ordinary_kriging(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    targets: ArrayLike,
    *,
    model_values: ArrayLike | None = None,
    radius: float | None = None,
    max_neighbors: int = 40,
    min_neighbors: int = 3,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, int | float]]:
    """Fit local ordinary kriging and estimate ``targets`` values.

    This is a one-shot convenience wrapper around :class:`Kriging` for callers
    who just want a single fit-and-interpolate call. Prefer instantiating
    ``Kriging`` directly when reusing the same fitted variogram across
    multiple ``interpolate`` calls, since this function always fits from
    scratch.

    Parameters
    ----------
    x, y, z : array_like
        Source coordinates and values.
    targets : array_like
        Target coordinates accepted by
        :meth:`apbase.kriging.Kriging.interpolate`.
    model_values : array_like or None, default None
        Optional native variogram model vector. If omitted, a variogram is
        fitted from ``x``, ``y``, and ``z``.
    radius : float or None, default None
        Local search radius. If ``None``, derive ``range / 3`` from the
        selected variogram.
    max_neighbors, min_neighbors : int
        Neighbor bounds used by the local search.
    return_diagnostics : bool, default False
        If ``True``, return interpolation diagnostics with the estimates.

    Returns
    -------
    numpy.ndarray or tuple[numpy.ndarray, dict]
        Estimates only, or ``(estimates, diagnostics)`` when
        ``return_diagnostics=True``.
    """
    kriging = Kriging(
        radius=radius,
        max_neighbors=max_neighbors,
        min_neighbors=min_neighbors,
        model_values=model_values,
    ).fit(x, y, z)
    estimates = kriging.interpolate(targets)

    if not return_diagnostics:
        return estimates

    radius_value = radius
    if radius_value is None:
        radius_value = float(kriging.model_params["range"]) / 3.0

    return estimates, {
        "radius": float(radius_value),
        "max_neighbors": int(max_neighbors),
        "min_neighbors": int(min_neighbors),
        "n_threads": int(kriging.n_threads),
    }
