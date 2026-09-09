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
    x, y:
        One-dimensional source coordinates. Required only when ``boundary``
        is omitted, to infer one as a concave hull of the finite pairs.
    resolution:
        Grid spacing in the same coordinate unit as ``x`` and ``y``.
    boundary:
        Optional Shapely geometry used as interpolation boundary. When given,
        ``x``/``y`` are not needed.
    chunk_size:
        Maximum number of candidate points tested per chunk.
    hull_ratio:
        Ratio used by Shapely concave hull when ``boundary`` is omitted.
    """
    return Grid(
        boundary=boundary,
        resolution=resolution,
        chunk_size=chunk_size,
        hull_ratio=hull_ratio,
    ).generate(x, y)
