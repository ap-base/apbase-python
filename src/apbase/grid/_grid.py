"""Regular grid generation for interpolation targets.

The Grid class builds a dense, regular point grid inside a supplied Shapely
boundary or inside a concave hull inferred from finite x/y source points. The
implementation chunks candidate rows to avoid materializing very large mesh
arrays at once. Used as a target-coordinate source by both IDW and kriging.
"""

from __future__ import annotations

import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import shapely
from numpy.typing import ArrayLike
from shapely.geometry.base import BaseGeometry

from apbase.common.arrays import as_float64_1d, filter_finite_xy, validate_same_size_xy
from apbase.common.coordinates import (
    CoordinateTransform,
    apply_metric_transform,
    guess_xy_coordinate_system,
    reproject_geographic_xy,
)
from apbase.config import resolve_grid_max_points_ceiling, resolve_n_threads

type Boundary = BaseGeometry

DEFAULT_CHUNK_SIZE = 1_000_000
DEFAULT_HULL_RATIO = 0.12

_HULL_DECIMATE_TARGET_POINTS = 20_000


def _coordinate_bucket(x: np.ndarray, y: np.ndarray) -> str:
    guess = guess_xy_coordinate_system(x, y, assume_finite=True)
    return "geographic" if guess == "geographic" else "metric"


class Grid:
    """Generate regular target points inside a boundary.

    The boundary either comes from the caller (``boundary=``, or the
    :meth:`from_wkt`/:meth:`from_wkb` constructors) or is inferred as a
    concave hull of ``x``/``y`` source points (:meth:`from_data_xy`, or
    plain ``fit(x, y)``) when none is given.

    Parameters
    ----------
    boundary:
        Optional Shapely geometry used as the interpolation domain. When
        omitted, a concave hull is inferred from the finite input points.
    resolution:
        Grid spacing in the same units as the input coordinates.
    chunk_size:
        Maximum number of candidate points tested per chunk.
    hull_ratio:
        Ratio passed to ``shapely.concave_hull`` when the boundary is
        inferred. Has no effect when ``boundary`` is supplied explicitly.
    coordinate_transform:
        Optional :class:`~apbase.common.coordinates.CoordinateTransform`
        applied to ``boundary`` before building the grid, so it lands in
        the same frame as ``x``/``y`` already converted with
        ``prepare_metric_xy``. Takes priority over ``geographic_mode``.
    geographic_mode:
        ``None`` (default), ``"local"``, or ``"utm"``. When ``boundary``
        looks geographic and ``x``/``y`` does not, reprojects ``boundary``
        to metric before building the grid.
    """

    __slots__ = (
        "boundary",
        "resolution",
        "chunk_size",
        "hull_ratio",
        "coordinate_transform",
        "geographic_mode",
        "_points",
        "_fitted_boundary",
        "_fitted",
    )

    def __init__(
        self,
        *,
        boundary: Boundary | None = None,
        resolution: float = 20.0,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        hull_ratio: float = DEFAULT_HULL_RATIO,
        coordinate_transform: CoordinateTransform | None = None,
        geographic_mode: str | None = None,
    ) -> None:
        self.boundary = boundary
        self.resolution = float(resolution)
        self.chunk_size = int(chunk_size)
        self.hull_ratio = float(hull_ratio)
        self.coordinate_transform = coordinate_transform
        self.geographic_mode = geographic_mode
        self._validate_config()

        self._points: np.ndarray | None = None
        self._fitted_boundary: Boundary | None = None
        self._fitted = False

    @classmethod
    def from_data_xy(cls, x: ArrayLike, y: ArrayLike, **kwargs: Any) -> Grid:
        """Create a grid and infer its boundary from x/y source points (concave hull).

        Equivalent to ``Grid(**kwargs).fit(x, y)`` with no explicit
        ``boundary``. Use :meth:`from_wkt`, :meth:`from_wkb`, or pass
        ``boundary=`` directly instead when the domain is already known --
        source points are not needed in that case.

        Parameters
        ----------
        x, y : array_like
            Source coordinates used to infer the concave-hull boundary.
        **kwargs
            Keyword arguments forwarded to :class:`Grid`.

        Returns
        -------
        Grid
            Fitted grid.
        """
        return cls(**kwargs).fit(x, y)

    @classmethod
    def from_wkt(cls, wkt: str, **kwargs: Any) -> Grid:
        """Create a grid whose boundary is parsed from a WKT string.

        No source points are needed: the grid is generated directly from the
        parsed geometry.

        Parameters
        ----------
        wkt : str
            Boundary geometry encoded as WKT.
        **kwargs
            Keyword arguments forwarded to :class:`Grid`.

        Returns
        -------
        Grid
            Fitted grid.
        """
        return cls(boundary=shapely.from_wkt(wkt), **kwargs).fit()

    @classmethod
    def from_wkb(cls, wkb: bytes, **kwargs: Any) -> Grid:
        """Create a grid whose boundary is parsed from WKB bytes.

        No source points are needed: the grid is generated directly from the
        parsed geometry.

        Parameters
        ----------
        wkb : bytes
            Boundary geometry encoded as WKB.
        **kwargs
            Keyword arguments forwarded to :class:`Grid`.

        Returns
        -------
        Grid
            Fitted grid.
        """
        return cls(boundary=shapely.from_wkb(wkb), **kwargs).fit()

    @property
    def is_fitted(self) -> bool:
        """Whether the grid points have already been generated.

        Returns
        -------
        bool
            ``True`` after :meth:`fit` succeeds.
        """
        return self._fitted

    @property
    def points(self) -> np.ndarray:
        """Generated target coordinates.

        Returns
        -------
        numpy.ndarray
            ``(n_points, 2)`` target coordinates inside the fitted boundary.
        """
        points, _ = self._get_fitted_state()
        return points

    @property
    def fitted_boundary(self) -> Boundary:
        """Prepared boundary used to keep target points inside the domain.

        Returns
        -------
        shapely.Geometry
            Buffered and prepared boundary used by point-in-boundary tests.
        """
        _, boundary = self._get_fitted_state()
        return boundary

    @property
    def n_threads(self) -> int:
        """Thread count used to parallelize boundary containment tests.

        Resolved live from ``apbase.config["n_threads"]`` (see the class
        docstring) on every access -- not cached at construction time, so it
        always reflects the current process-wide setting.

        Returns
        -------
        int
            Current thread count.
        """
        return resolve_n_threads()

    def fit(self, x: ArrayLike | None = None, y: ArrayLike | None = None) -> Grid:
        """Generate grid points.

        When ``boundary`` was supplied to the constructor, the grid is built
        directly from it; ``x``/``y`` are optional in that case, used only to
        reconcile ``boundary`` against ``coordinate_transform``/
        ``geographic_mode`` (see the class docstring) -- ``x``/``y`` are never
        used to build the grid points themselves when ``boundary`` is given.
        Otherwise ``x``/``y`` are required: non-finite pairs are removed and
        a concave hull inferred from the rest becomes the boundary. Input
        coordinates are assumed to already be in the target projected
        coordinate system, unless converted via ``coordinate_transform``/
        ``geographic_mode``.

        Parameters
        ----------
        x, y : array_like or None, default None
            Source coordinates used only when inferring a boundary or
            reconciling a provided boundary against a coordinate transform.

        Returns
        -------
        Grid
            This grid with generated ``points`` and prepared
            ``fitted_boundary``.

        Raises
        ------
        ValueError
            If no boundary can be inferred, the generated candidate grid is
            empty, or the candidate grid exceeds
            ``apbase.config["grid_max_points_ceiling"]``.
        """
        data_xy: tuple[np.ndarray, np.ndarray] | None = None
        if self.boundary is not None:
            raw_boundary = self.boundary
            if x is not None and y is not None:
                x_array = as_float64_1d(x, "x")
                y_array = as_float64_1d(y, "y")
                validate_same_size_xy(x_array, y_array)
                data_xy = filter_finite_xy(x_array, y_array)
                if data_xy[0].size > 0:
                    raw_boundary = self._reconcile_boundary_crs(raw_boundary, data_xy[0], data_xy[1])
            boundary = self._prepare_boundary(raw_boundary)
        else:
            if x is None or y is None:
                raise ValueError("x and y are required to infer a boundary when none is provided")
            x_array = as_float64_1d(x, "x")
            y_array = as_float64_1d(y, "y")
            validate_same_size_xy(x_array, y_array)

            x_valid, y_valid = filter_finite_xy(x_array, y_array)
            if x_valid.size == 0:
                raise ValueError("grid requires at least one finite point")

            data_xy = (x_valid, y_valid)
            boundary = self._infer_boundary(x_valid, y_valid)

        points = self._generate_points(boundary)

        self._points = points
        self._fitted_boundary = boundary
        self._fitted = True
        return self

    def generate(self, x: ArrayLike | None = None, y: ArrayLike | None = None) -> np.ndarray:
        """Fit the grid and return generated target coordinates.

        Parameters
        ----------
        x, y : array_like or None, default None
            Source coordinates forwarded to :meth:`fit`.

        Returns
        -------
        numpy.ndarray
            ``(n_points, 2)`` target coordinates inside the boundary.
        """
        return self.fit(x, y).points

    def __call__(self, x: ArrayLike | None = None, y: ArrayLike | None = None) -> np.ndarray:
        """Fit the grid and return generated target coordinates.

        Parameters
        ----------
        x, y : array_like or None, default None
            Source coordinates forwarded to :meth:`fit`.

        Returns
        -------
        numpy.ndarray
            ``(n_points, 2)`` target coordinates inside the boundary.
        """
        return self.generate(x, y)

    def _infer_boundary(self, x: np.ndarray, y: np.ndarray) -> Boundary:
        hull_x, hull_y = self._decimate_for_hull(x, y)
        boundary = shapely.concave_hull(
            shapely.multipoints(np.column_stack((hull_x, hull_y))),
            ratio=self.hull_ratio,
        )
        return self._prepare_boundary(boundary)

    def _prepare_boundary(self, boundary: Boundary) -> Boundary:
        if boundary.is_empty:
            raise ValueError("the resulting boundary is empty")

        buffered_boundary = shapely.buffer(boundary, self.resolution * 0.5)
        if buffered_boundary.is_empty:
            raise ValueError("the buffered boundary is empty")

        shapely.prepare(buffered_boundary)
        return buffered_boundary

    def _reconcile_boundary_crs(self, boundary: Boundary, x: np.ndarray, y: np.ndarray) -> Boundary:
        if self.coordinate_transform is not None:
            return shapely.transform(
                boundary,
                lambda coords: np.column_stack(
                    apply_metric_transform(coords[:, 0], coords[:, 1], self.coordinate_transform)
                ),
            )

        if self.geographic_mode is not None:
            boundary_minx, boundary_miny, boundary_maxx, boundary_maxy = boundary.bounds
            boundary_bucket = _coordinate_bucket(
                np.array([boundary_minx, boundary_maxx]),
                np.array([boundary_miny, boundary_maxy]),
            )
            data_bucket = _coordinate_bucket(x, y)
            if boundary_bucket == "geographic" and data_bucket != "geographic":
                mode = self.geographic_mode
                return shapely.transform(
                    boundary,
                    lambda coords: np.column_stack(
                        reproject_geographic_xy(coords[:, 0], coords[:, 1], mode)
                    ),
                )

        self._warn_if_boundary_crs_mismatch(boundary, x, y)
        return boundary

    @staticmethod
    def _warn_if_boundary_crs_mismatch(boundary: Boundary, x: np.ndarray, y: np.ndarray) -> None:
        boundary_minx, boundary_miny, boundary_maxx, boundary_maxy = boundary.bounds
        data_bucket = _coordinate_bucket(x, y)
        boundary_bucket = _coordinate_bucket(
            np.array([boundary_minx, boundary_maxx]),
            np.array([boundary_miny, boundary_maxy]),
        )
        if data_bucket != boundary_bucket:
            warnings.warn(
                "Grid boundary and input x/y coordinates appear to be in different "
                "coordinate systems (one looks geographic/lon-lat, the other looks "
                "projected/metric). Grid.resolution is interpreted in the same unit "
                "as x/y, so this mismatch likely means resolution and/or area are "
                "wrong. Make sure boundary and x/y share the same CRS (see "
                "apbase.common.coordinates.prepare_metric_xy).",
                UserWarning,
                stacklevel=3,
            )

    @staticmethod
    def _axis_range(lo: float, hi: float, step: float) -> np.ndarray:
        # np.linspace with a count derived via round(), rather than np.arange
        # with a float step, avoids floating-point accumulation error that
        # can silently drop or add a point at the far edge of the axis.
        start = np.floor(lo / step) * step
        stop = np.ceil(hi / step) * step
        count = int(round((stop - start) / step)) + 1
        if count <= 0:
            return np.empty(0, dtype=np.float64)
        return np.linspace(start, stop, count, dtype=np.float64)

    def _generate_points(self, boundary: Boundary) -> np.ndarray:
        min_x, min_y, max_x, max_y = boundary.bounds
        step = self.resolution

        x_axis = self._axis_range(min_x, max_x, step)
        y_axis = self._axis_range(min_y, max_y, step)

        nx = x_axis.size
        if nx == 0:
            raise ValueError("empty grid on the x axis")
        ny = y_axis.size
        if ny == 0:
            raise ValueError("empty grid on the y axis")

        candidate_count = nx * ny
        ceiling = resolve_grid_max_points_ceiling()
        if candidate_count > ceiling:
            raise ValueError(
                f"grid would generate {candidate_count:,} candidate points ({nx:,} x {ny:,}) "
                f"before boundary filtering, above the {ceiling:,} ceiling "
                '(apbase.config["grid_max_points_ceiling"]) -- check that `resolution` and '
                "`boundary` share the same unit/CRS; a common cause is geographic input "
                "auto-detected and reprojected to UTM meters while `resolution` was sized for a "
                "much coarser local unit"
            )

        rows_per_chunk = max(1, self.chunk_size // nx)
        blocks = self._candidate_blocks(x_axis, y_axis, rows_per_chunk)
        n_threads = resolve_n_threads()

        x_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        if n_threads > 1:
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                for candidate_x, candidate_y, mask in self._intersects_blocks_parallel(
                    executor, boundary, blocks, n_threads
                ):
                    if not np.any(mask):
                        continue
                    x_parts.append(candidate_x[mask])
                    y_parts.append(candidate_y[mask])
        else:
            for candidate_x, candidate_y in blocks:
                mask = shapely.intersects_xy(boundary, candidate_x, candidate_y)
                if not np.any(mask):
                    continue
                x_parts.append(candidate_x[mask])
                y_parts.append(candidate_y[mask])

        if not x_parts:
            raise ValueError("no grid point falls inside the boundary")

        output_size = sum(part.size for part in x_parts)
        points = np.empty((output_size, 2), dtype=np.float64, order="F")
        points[:, 0] = x_parts[0] if len(x_parts) == 1 else np.concatenate(x_parts)
        points[:, 1] = y_parts[0] if len(y_parts) == 1 else np.concatenate(y_parts)

        return points

    @staticmethod
    def _intersects_blocks_parallel(executor, boundary, blocks, n_threads):
        """Run ``shapely.intersects_xy`` per block on a thread pool, in order.

        ``shapely.intersects_xy`` releases the GIL, so this parallelizes real
        CPU work across ``n_threads`` threads. Uses a bounded sliding window
        (at most ``n_threads`` chunks in flight) instead of ``executor.map``
        -- which would eagerly submit every chunk up front -- to preserve the
        same peak-memory bound ``chunk_size`` chunking already provides.
        Yields results in the same row-major order as ``blocks``, which
        downstream code (``MapResult.to_raster``) relies on.
        """
        block_iter = iter(blocks)
        window: deque[tuple[np.ndarray, np.ndarray, Any]] = deque()

        for _ in range(n_threads):
            try:
                candidate_x, candidate_y = next(block_iter)
            except StopIteration:
                break
            future = executor.submit(shapely.intersects_xy, boundary, candidate_x, candidate_y)
            window.append((candidate_x, candidate_y, future))

        while window:
            candidate_x, candidate_y, future = window.popleft()
            try:
                next_x, next_y = next(block_iter)
            except StopIteration:
                pass
            else:
                next_future = executor.submit(shapely.intersects_xy, boundary, next_x, next_y)
                window.append((next_x, next_y, next_future))
            yield candidate_x, candidate_y, future.result()

    @staticmethod
    def _decimate_for_hull(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = x.size
        if n <= _HULL_DECIMATE_TARGET_POINTS:
            return x, y

        # Cell size derived from the point cloud's own density (bounding-box
        # area / target point count), not from `resolution`: the two are
        # unrelated in general (dense source data onto a coarse grid vs.
        # sparse data onto a fine grid), so a resolution-derived cell can
        # land far below the actual point spacing and decimate nothing while
        # still paying its own overhead. This self-calibrates regardless of
        # how resolution and source density relate.
        area = float(x.max() - x.min()) * float(y.max() - y.min())
        if not np.isfinite(area) or area <= 0.0:
            return x, y

        cell = np.sqrt(area / _HULL_DECIMATE_TARGET_POINTS)
        if not np.isfinite(cell) or cell <= 0.0:
            return x, y

        inv_cell = 1.0 / cell
        ix = np.floor(x * inv_cell).astype(np.int64)
        iy = np.floor(y * inv_cell).astype(np.int64)
        ix -= ix.min()
        iy -= iy.min()
        keys = ix * (iy.max() + 1) + iy

        _, first_positions = np.unique(keys, return_index=True)
        if first_positions.size == n:
            return x, y
        return x[first_positions], y[first_positions]

    @staticmethod
    def _candidate_blocks(
        x_axis: np.ndarray,
        y_axis: np.ndarray,
        rows_per_chunk: int,
    ):
        nx = x_axis.size
        for start in range(0, y_axis.size, rows_per_chunk):
            y_block = y_axis[start : start + rows_per_chunk]
            yield (
                np.tile(x_axis, y_block.size),
                np.repeat(y_block, nx),
            )

    def _get_fitted_state(self) -> tuple[np.ndarray, Boundary]:
        if not self._fitted or self._points is None or self._fitted_boundary is None:
            raise ValueError("call fit() before accessing the grid")
        return self._points, self._fitted_boundary

    def _validate_config(self) -> None:
        if not np.isfinite(self.resolution) or self.resolution <= 0.0:
            raise ValueError("resolution must be finite and greater than zero")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if not np.isfinite(self.hull_ratio) or self.hull_ratio < 0.0 or self.hull_ratio > 1.0:
            raise ValueError("hull_ratio must be between 0 and 1")


__all__ = ["Grid"]
