"""Regular target-grid generation, shared by IDW and kriging interpolation.

This package exports the ``Grid`` class and a functional convenience wrapper
for generating regular point grids used as interpolation targets.
"""

from __future__ import annotations

from numpy import ndarray
from numpy.typing import ArrayLike

from ._grid import DEFAULT_CHUNK_SIZE, DEFAULT_HULL_RATIO, Grid

__all__ = ["Grid", "create_point_grid"]


def create_point_grid(
    *,
    x: ArrayLike | None = None,
    y: ArrayLike | None = None,
    resolution: float,
    boundary: object | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    hull_ratio: float = DEFAULT_HULL_RATIO,
) -> ndarray:
    """Create regular target coordinates, from a boundary or from ``x``/``y``.

    Parameters
    ----------
    x, y : array_like or None, default None
        One-dimensional source coordinates. Required only when ``boundary``
        is omitted, to infer one as a concave hull of the finite pairs.
    resolution : float
        Grid spacing in the same coordinate unit as ``x`` and ``y``.
    boundary : object or None, default None
        Optional Shapely geometry used as interpolation boundary. When given,
        ``x``/``y`` are not needed.
    chunk_size : int, default DEFAULT_CHUNK_SIZE
        Maximum number of candidate points tested per chunk.
    hull_ratio : float, default DEFAULT_HULL_RATIO
        Ratio used by Shapely concave hull when ``boundary`` is omitted.

    Returns
    -------
    numpy.ndarray
        ``(n_points, 2)`` generated target coordinates inside the boundary.

    Raises
    ------
    ValueError
        If ``boundary`` is omitted and ``x``/``y`` cannot infer a valid
        boundary, or if the candidate grid is empty or too large.
    """
    return Grid(
        boundary=boundary,
        resolution=resolution,
        chunk_size=chunk_size,
        hull_ratio=hull_ratio,
    ).generate(x, y)
