"""Automatic high-performance map generation.

``Map`` is the primary entry point. It orchestrates the existing building
blocks (``SpatialFilter``, ``Grid``, ``Variogram``, ``cross_validate``,
``IDW``, ``Kriging``) behind one map-building pipeline. By default
(``geographic_mode=None``), ``x``/``y`` are checked automatically
(:func:`apbase.common.coordinates.guess_xy_coordinate_system`): geographic
(lon/lat) input is detected and converted internally via
:func:`apbase.common.coordinates.prepare_metric_xy`
-- itself auto-picking ``"local"`` vs ``"utm"`` from the input's own extent
-- before the pipeline runs, and the result coordinates are converted back
to the original input metric automatically via
:func:`apbase.common.coordinates.restore_original_xy`; already-metric/projected
input is left untouched, exactly like ``SpatialFilter``/``Grid``/``IDW``/
``Kriging`` already assume. Callers never touch ``prepare_metric_xy``/
``restore_original_xy`` directly. Passing ``geographic_mode="local"`` or
``"utm"`` explicitly forces that projection instead of auto-picking it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import shapely
from numpy.typing import ArrayLike
from shapely.geometry.base import BaseGeometry

from apbase.common.arrays import (
    as_float64_1d,
    filter_finite_xyz,
    normalize_masked_regular_grid_spec,
    validate_same_size_xyz,
)
from apbase.common.coordinates import (
    CoordinateTransform,
    guess_xy_coordinate_system,
    prepare_metric_xy,
    restore_original_xy,
)
from apbase.config import resolve_variogram_profile
from apbase.cross_validate import CrossValidationResult, cross_validate, select_best_model
from apbase.filtering import FilterStatistics, SpatialFilter
from apbase.grid import Grid
from apbase.idw import IDW
from apbase.kriging import Kriging
from apbase.variogram import Variogram

type Bounds = BaseGeometry | str | bytes


@dataclass(frozen=True)
class MapResult:
    """Result of automatic map generation.

    ``x``, ``y``, ``z`` are a compact point list -- one row per target point
    inside the boundary, no padding for cells outside it. Call
    :meth:`to_raster`/:meth:`to_array3d` to materialize a padded rectangular
    array, only when one is actually needed.

    Attributes
    ----------
    x, y, z : numpy.ndarray
        Compact target coordinates and interpolated values.
    method : {"idw", "kriging"}
        Interpolation method selected by cross-validation.
    cross_validation : CrossValidationResult
        Leave-one-out diagnostics used for method selection.
    radius : float
        Local search radius used by the final interpolation.
    resolution : float
        Output grid spacing.
    n_source_points : int
        Number of source points used after finite filtering and optional
        spatial filtering.
    filter_statistics : FilterStatistics or None
        Local filtering diagnostics, or ``None`` when filtering was disabled.
    coordinate_transform : CoordinateTransform or None
        Transform used to recover output coordinates in the original input
        coordinate system.
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    method: Literal["idw", "kriging"]
    cross_validation: CrossValidationResult
    radius: float
    resolution: float
    n_source_points: int
    filter_statistics: FilterStatistics | None
    coordinate_transform: CoordinateTransform | None

    def to_raster(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reconstruct ``(X, Y, Z)`` as ``(ny, nx)`` arrays, NaN outside the boundary.

        ``X``/``Y`` hold the real input coordinates (same unit as the source
        ``x``/``y``/``bounds``), not a rebased ``0..nx``/``0..ny`` index.

        When ``coordinate_transform.geographic_mode == "utm"`` (i.e. the map
        was built with ``geographic_mode="utm"`` from detected geographic
        input), the inverse UTM->lon/lat reprojection is non-linear, so a grid
        that is exactly regular internally may no longer be exactly regular
        in lon/lat -- this can raise ``ValueError`` for non-trivial extents.
        ``geographic_mode="local"`` uses a fixed affine transform instead and
        is always safe for ``to_raster``/``to_array3d``.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
            Rectangular ``X``, ``Y``, and ``Z`` arrays. Masked-out cells are
            ``NaN``.

        Raises
        ------
        ValueError
            If compact points cannot be represented as a regular masked grid.
        """
        spec = normalize_masked_regular_grid_spec(np.column_stack((self.x, self.y)))
        if spec is None:
            raise ValueError("map points do not form a regular grid; cannot build a raster")

        shape = (spec.full_ny, spec.full_nx)
        x_out = np.full(shape, np.nan, dtype=np.float64)
        y_out = np.full(shape, np.nan, dtype=np.float64)
        z_out = np.full(shape, np.nan, dtype=np.float64)

        pos = 0
        for row, x_start, count in zip(spec.row_indices, spec.x_start_indices, spec.counts, strict=True):
            row = int(row)
            x_start = int(x_start)
            count = int(count)
            cols = slice(x_start, x_start + count)
            x_out[row, cols] = spec.min_x + np.arange(x_start, x_start + count, dtype=np.float64) * spec.dx
            y_out[row, cols] = spec.min_y + row * spec.dy
            z_out[row, cols] = self.z[pos : pos + count]
            pos += count

        return x_out, y_out, z_out

    def to_array3d(self) -> np.ndarray:
        """Stack :meth:`to_raster` into a single ``(ny, nx, 3)`` array.

        Returns
        -------
        numpy.ndarray
            Rectangular array whose last axis stores ``X``, ``Y``, and ``Z``.
        """
        return np.dstack(self.to_raster())


def create_map(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    *,
    resolution: float,
    filter: bool = True,
    bounds: Bounds | None = None,
    geographic_mode: str | None = None,
    power: float = 2.0,
    max_neighbors: int = 40,
    min_neighbors: int = 3,
    max_cv_points: int = 500,
    seed: int | None = None,
    filter_kwargs: dict[str, Any] | None = None,
) -> MapResult:
    """Functional shortcut for the automatic :class:`Map` pipeline.

    Filters, cross-validates, and interpolates ``x, y, z`` into a map. Picks
    IDW or ordinary kriging automatically via
    :func:`~apbase.cross_validate.select_best_model` on the leave-one-out
    results from :func:`~apbase.cross_validate.cross_validate`, then
    interpolates onto a regular grid built from ``bounds`` (or a concave
    hull inferred from ``x``/``y`` when omitted).

    By default (``geographic_mode=None``), ``x``/``y`` are checked
    automatically (:func:`apbase.common.coordinates.guess_xy_coordinate_system`):
    already projected/metric input is used as-is, with no CRS conversion --
    the same behavior as before. Geographic (lon/lat) input is detected and
    handled automatically instead: it -- and ``bounds``, when given -- is
    converted to metric coordinates internally
    (:func:`apbase.common.coordinates.prepare_metric_xy`, which
    itself auto-picks ``"local"`` vs ``"utm"`` from the input's own extent
    against ``apbase.config["local_mode_extent_km_ceiling"]``) before the
    pipeline runs, and ``result.x``/``result.y`` are converted back to the
    original input metric automatically; you never call
    ``prepare_metric_xy``/``restore_original_xy`` yourself. Passing
    ``geographic_mode="local"`` or ``"utm"`` explicitly forces that
    projection instead of auto-picking it (still only applied to input
    actually detected as geographic).

    Parameters
    ----------
    x, y, z:
        One-dimensional source coordinates and values.
    resolution:
        Output grid spacing. In the metric unit of ``x``/``y`` when they are
        not geographic; in meters (real UTM or the spherical local
        approximation) when geographic input is detected, whether
        ``geographic_mode`` was left at ``None`` (auto-picked) or set
        explicitly.
    filter:
        Whether to run :class:`~apbase.filtering.SpatialFilter` on the source
        data before cross-validating and interpolating.
    bounds:
        Optional interpolation domain: a Shapely geometry, a WKT string, or
        WKB bytes, in the same unit as ``x``/``y`` (before any geographic
        conversion). When omitted, a concave hull is inferred from the
        (filtered) source points.
    geographic_mode:
        ``None`` (default): auto-detect, as described above -- geographic
        input auto-picks ``"local"`` or ``"utm"``; already-metric input is
        unaffected. ``"local"`` or ``"utm"``: force that projection for
        detected geographic input, using the same semantics as
        :func:`apbase.common.coordinates.prepare_metric_xy`'s
        ``geographic_mode``. ``"local"`` is always compatible with
        :meth:`MapResult.to_raster`; ``"utm"``'s inverse reprojection is
        non-linear and can break exact grid regularity for non-trivial
        extents -- see :meth:`MapResult.to_raster`.
    power:
        IDW distance exponent, used only if IDW is selected.
    max_neighbors, min_neighbors:
        Neighbor bounds shared by cross-validation and the final IDW/kriging
        fit.
    max_cv_points, seed:
        Forwarded to :func:`~apbase.cross_validate.cross_validate`.
    filter_kwargs:
        Extra keyword arguments forwarded to
        :class:`~apbase.filtering.SpatialFilter` (ignored when
        ``filter=False``).

    Returns
    -------
    MapResult
        Compact map output plus cross-validation and filtering diagnostics.

    Examples
    --------
    >>> result = create_map(x, y, z, resolution=10.0)
    >>> result.method
    'kriging'
    """
    return _build_map(
        x,
        y,
        z,
        resolution=resolution,
        filter=filter,
        bounds=bounds,
        geographic_mode=geographic_mode,
        power=power,
        max_neighbors=max_neighbors,
        min_neighbors=min_neighbors,
        max_cv_points=max_cv_points,
        seed=seed,
        filter_kwargs=filter_kwargs,
    )


class Map:
    """Primary automatic map builder.

    ``Map`` is the recommended high-level entry point for production map
    generation. It filters source data, builds the target grid, fits one
    shared variogram, cross-validates IDW against ordinary kriging, selects
    the statistically better method, and interpolates the final map.

    Configure it once and call it with source data::

        builder = Map(resolution=10.0, bounds=boundary)
        result = builder(x, y, z)

    The same configured instance can be reused across datasets::

        result_a = builder.fit_generate(x_a, y_a, z_a)
        result_b = builder(x_b, y_b, z_b)

    For one-shot functional usage, call :func:`create_map`.

    Parameters
    ----------
    x, y, z : array_like or None, default None
        Optional source coordinates and values. When ``x`` is supplied, all
        three must be supplied and the map is generated during construction.
    resolution : float
        Output grid spacing.
    filter : bool, default True
        Whether to apply :class:`~apbase.filtering.SpatialFilter`.
    bounds : geometry, str, bytes, or None, default None
        Optional Shapely geometry, WKT, or WKB interpolation domain.
    geographic_mode : {"local", "utm"} or None, default None
        Geographic projection mode passed through to :func:`create_map`.
    power : float, default 2.0
        IDW distance exponent.
    max_neighbors, min_neighbors : int
        Neighbor bounds shared by cross-validation and final interpolation.
    max_cv_points : int, default 500
        Maximum leave-one-out samples used by cross-validation.
    seed : int or None, default None
        Random seed for cross-validation sampling and model selection.
    filter_kwargs : dict or None, default None
        Extra keyword arguments forwarded to
        :class:`~apbase.filtering.SpatialFilter`.
    """

    __slots__ = (
        "resolution",
        "filter",
        "bounds",
        "geographic_mode",
        "power",
        "max_neighbors",
        "min_neighbors",
        "max_cv_points",
        "seed",
        "filter_kwargs",
        "result",
    )

    def __init__(
        self,
        x: ArrayLike | None = None,
        y: ArrayLike | None = None,
        z: ArrayLike | None = None,
        *,
        resolution: float,
        filter: bool = True,
        bounds: Bounds | None = None,
        geographic_mode: str | None = None,
        power: float = 2.0,
        max_neighbors: int = 40,
        min_neighbors: int = 3,
        max_cv_points: int = 500,
        seed: int | None = None,
        filter_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.resolution = resolution
        self.filter = filter
        self.bounds = bounds
        self.geographic_mode = geographic_mode
        self.power = power
        self.max_neighbors = max_neighbors
        self.min_neighbors = min_neighbors
        self.max_cv_points = max_cv_points
        self.seed = seed
        self.filter_kwargs = filter_kwargs
        self.result: MapResult | None = None

        if x is not None:
            if y is None or z is None:
                raise ValueError("x, y, and z must be provided together, or all omitted")
            self.fit_generate(x, y, z)

    def fit_generate(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> MapResult:
        """Run the full pipeline and store the generated map.

        Parameters
        ----------
        x, y, z : array_like
            Source coordinates and values.

        Returns
        -------
        MapResult
            Generated map result, also stored on :attr:`result`.
        """
        self.result = _build_map(
            x,
            y,
            z,
            resolution=self.resolution,
            filter=self.filter,
            bounds=self.bounds,
            geographic_mode=self.geographic_mode,
            power=self.power,
            max_neighbors=self.max_neighbors,
            min_neighbors=self.min_neighbors,
            max_cv_points=self.max_cv_points,
            seed=self.seed,
            filter_kwargs=self.filter_kwargs,
        )
        return self.result

    def __call__(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> MapResult:
        """Run the full pipeline and return the generated map.

        Parameters
        ----------
        x, y, z : array_like
            Source coordinates and values.

        Returns
        -------
        MapResult
            Generated map result.
        """
        return self.fit_generate(x, y, z)


def _build_map(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    *,
    resolution: float,
    filter: bool,
    bounds: Bounds | None,
    geographic_mode: str | None,
    power: float,
    max_neighbors: int,
    min_neighbors: int,
    max_cv_points: int,
    seed: int | None,
    filter_kwargs: dict[str, Any] | None,
) -> MapResult:
    x_array = as_float64_1d(x, "x")
    y_array = as_float64_1d(y, "y")
    z_array = as_float64_1d(z, "z")
    validate_same_size_xyz(x_array, y_array, z_array)

    transform: CoordinateTransform | None = None
    if geographic_mode is not None or guess_xy_coordinate_system(x_array, y_array) == "geographic":
        x_array, y_array, _, _, _, _, transform = prepare_metric_xy(
            x_array,
            y_array,
            x_array,
            y_array,
            geographic_mode=geographic_mode,
            return_transform=True,
        )

    x_array, y_array, z_array = filter_finite_xyz(x_array, y_array, z_array)

    filter_statistics: FilterStatistics | None = None
    if filter:
        spatial_filter = SpatialFilter(**(filter_kwargs or {}))
        x_array, y_array, z_array = spatial_filter.fit_filter(x_array, y_array, z_array)
        filter_statistics = spatial_filter.statistics

    n_lags, max_pairs, max_distance = resolve_variogram_profile()
    variogram = Variogram(
        n_lags=n_lags,
        max_pairs=max_pairs,
        max_distance=max_distance,
    ).fit(x_array, y_array, z_array)
    radius = float(variogram.model_params["range"]) / 3.0

    cv_result = cross_validate(
        x_array,
        y_array,
        z_array,
        max_points=max_cv_points,
        seed=seed,
        radius=radius,
        power=power,
        max_neighbors=max_neighbors,
        min_neighbors=min_neighbors,
        model_values=variogram.model_values,
    )
    method: Literal["idw", "kriging"] = select_best_model(cv_result, seed=seed).method

    boundary = _normalize_bounds(bounds)
    grid = Grid(boundary=boundary, resolution=resolution, coordinate_transform=transform).fit(
        x_array, y_array
    )

    if method == "idw":
        estimates = (
            IDW(radius=radius, power=power, max_neighbors=max_neighbors, min_neighbors=min_neighbors)
            .fit(x_array, y_array, z_array)
            .interpolate(grid.points)
        )
    else:
        estimates = (
            Kriging(
                radius=radius,
                model_values=variogram.model_values,
                max_neighbors=max_neighbors,
                min_neighbors=min_neighbors,
            )
            .fit(x_array, y_array, z_array)
            .interpolate(grid.points)
        )

    if transform is not None:
        result_x, result_y = restore_original_xy(grid.points[:, 0], grid.points[:, 1], transform)
    else:
        result_x = np.ascontiguousarray(grid.points[:, 0])
        result_y = np.ascontiguousarray(grid.points[:, 1])

    return MapResult(
        x=result_x,
        y=result_y,
        z=estimates,
        method=method,
        cross_validation=cv_result,
        radius=radius,
        resolution=float(resolution),
        n_source_points=int(x_array.size),
        filter_statistics=filter_statistics,
        coordinate_transform=transform,
    )


def _normalize_bounds(bounds: Bounds | None) -> BaseGeometry | None:
    if bounds is None or isinstance(bounds, BaseGeometry):
        return bounds
    if isinstance(bounds, str):
        return shapely.from_wkt(bounds)
    if isinstance(bounds, bytes):
        return shapely.from_wkb(bounds)
    raise ValueError("bounds must be a Shapely geometry, a WKT string, or WKB bytes")


__all__ = ["Map", "MapResult", "create_map"]
