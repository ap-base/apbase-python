"""High-level cokriging API.

Estimates a primary variable using one or more spatially correlated
secondary variables. Run :func:`apbase.cross_variogram.
screen_secondary_variables` first to select admissible secondaries.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ._interpolation import CoKriging, SecondaryInput

__all__ = ["CoKriging", "co_kriging"]


def co_kriging(
    x0: ArrayLike,
    y0: ArrayLike,
    z0: ArrayLike,
    secondaries: SecondaryInput,
    targets: ArrayLike,
    *,
    method: str = "collocated",
    model_values: ArrayLike | None = None,
    radius: float | None = None,
    max_neighbors: int = 40,
    min_neighbors: int = 3,
) -> np.ndarray:
    """Fit cokriging and estimate ``targets`` values in one call.

    One-shot convenience wrapper around :class:`CoKriging` for callers who
    just want a single fit-and-interpolate call. Prefer instantiating
    ``CoKriging`` directly when reusing the same fitted model across
    multiple ``interpolate`` calls, since this function always fits from
    scratch.
    """
    return (
        CoKriging(
            method=method,
            radius=radius,
            max_neighbors=max_neighbors,
            min_neighbors=min_neighbors,
            model_values=model_values,
        )
        .fit(x0, y0, z0, secondaries)
        .interpolate(targets)
    )
