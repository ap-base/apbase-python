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
    max_colocation_dist: float | None = None,
    max_secondary_neighbors: int | None = None,
) -> np.ndarray:
    """Fit cokriging and estimate ``targets`` values in one call.

    One-shot convenience wrapper around :class:`CoKriging` for callers who
    just want a single fit-and-interpolate call. Prefer instantiating
    ``CoKriging`` directly when reusing the same fitted model across
    multiple ``interpolate`` calls, since this function always fits from
    scratch.

    Parameters
    ----------
    x0, y0, z0 : array_like
        Primary variable source coordinates and values.
    secondaries : mapping or sequence
        Secondary variables accepted by :meth:`apbase.cokriging.CoKriging.fit`.
    targets : array_like
        Target coordinates accepted by
        :meth:`apbase.cokriging.CoKriging.interpolate`.
    method : {"collocated", "icm", "lmc"}, default "collocated"
        Cokriging method.
    model_values : array_like or None, default None
        Optional pre-fitted coregionalization model vector, currently
        supported only for ``method="collocated"``.
    radius : float or None, default None
        Local search radius. If ``None``, derive ``range`` from the
        primary variogram (``range / 3`` with 10 000 or more points).
    max_neighbors, min_neighbors : int
        Primary neighbor bounds used by the local search.
    max_colocation_dist : float or None, default None
        Distance within which a primary sample is paired with its nearest
        secondary point to estimate the cross-covariances; ``None`` pairs
        only exactly coincident locations. Heterotopic secondaries, such as
        rasters, need a distance (e.g. half the cell diagonal).
    max_secondary_neighbors : int or None, default None
        ``"icm"``/``"lmc"`` only: maximum neighbors of each secondary per
        target; ``None`` uses ``max_neighbors``.

    Returns
    -------
    numpy.ndarray
        One cokriging estimate per target coordinate.
    """
    return (
        CoKriging(
            method=method,
            radius=radius,
            max_neighbors=max_neighbors,
            min_neighbors=min_neighbors,
            model_values=model_values,
            max_colocation_dist=max_colocation_dist,
            max_secondary_neighbors=max_secondary_neighbors,
        )
        .fit(x0, y0, z0, secondaries)
        .interpolate(targets)
    )
